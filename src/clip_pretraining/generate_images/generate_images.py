import torch
import pdb
import time
from diffusers import FluxPipeline
from accelerate import PartialState

import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import math
import shutil
import datetime
import yaml

def get_model():
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    return pipe

def get_batched_data(args):
    # cols: imagenet id, prompt, word pair ID
    prompts_df = pd.read_csv(args.prompts_file, comment="#")
    if 'orig_caption' in prompts_df.columns:
        prompts_df['caption'] = prompts_df['caption'].fillna(prompts_df['orig_caption'])
    num_batches = math.ceil(len(prompts_df) / args.batch_size)
    batches = []
    for i in range(num_batches):
        df_idx = i*args.batch_size
        chunk = prompts_df.iloc[df_idx:df_idx+args.batch_size]
        batch = {
            "word_pair_id": chunk.word_pair_id.tolist(),
            "imagenet_id": chunk.assigned_label_int.tolist(),
            "prompts": chunk.caption.tolist()
        }
        batches.append(batch)

    return batches

def distprint(content, state):
    if state.is_main_process:
        print(content)

distributed_state = PartialState()

parser = argparse.ArgumentParser()
parser.add_argument("--prompts_file", type=str)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--outdir", type=str, help="path to save images")
parser.add_argument("--guidance_scale", type=float, default=3.5)
args = parser.parse_args()
distprint(vars(args), distributed_state)

pipe = get_model()
batched_prompts = get_batched_data(args)
distprint(f"total batches: {len(batched_prompts)}", distributed_state)

outdir = Path(args.outdir)

if distributed_state.is_main_process:
    if (outdir / "config.yml").exists():
        shutil.rmtree(outdir)
        print("outdir already exists, deleting...")
    outdir.mkdir(parents=True)
    # Write args to config file for reproducibility
    config_path = outdir / "config.yml"
    with open(config_path, "w") as f:
        yaml.dump(vars(args), f)
    print(f"saving images to {outdir}")

with distributed_state.split_between_processes(batched_prompts) as dist_batches:
    if distributed_state.is_main_process:
        pbar = tqdm(total=len(dist_batches), desc="Dataset progress")
        # DEBUGGING
        print(f"num distributed batches: {len(dist_batches)}")

    pipe.to(distributed_state.device)
    for i, batch in enumerate(dist_batches):
        if distributed_state.is_main_process:
            file_count = sum(1 for _ in outdir.rglob("*") if _.is_file())
            print(f"total files: {file_count}, expected: {i * args.batch_size * distributed_state.num_processes}", flush=True)

        images = pipe(
            batch['prompts'],
            height=512,
            width=512,
            guidance_scale=5.0,
            num_inference_steps=28,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0) # TODO: what's this
        ).images
        distprint(f"generated {len(images)} images", distributed_state)

        for j, image in enumerate(images):
            label_dir = outdir / str(batch['imagenet_id'][j]).zfill(3)
            if not label_dir.exists():
                try:
                    label_dir.mkdir()
                except FileExistsError as e:
                    print(f"error creating label dir {label_dir}: {e}, trying in 5 seconds...")
                    time.sleep(5)
                    if not label_dir.exists():
                        label_dir.mkdir()

            image_path = label_dir / f"{batch['word_pair_id'][j]}.png"
            if j == 0:
                distprint(f"saving image to {image_path}", distributed_state)

            image.save(image_path)

        if distributed_state.is_main_process:
            pbar.update(1)
            
            if i % 10 == 0:
                rate = pbar.format_dict["rate"]
                remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0 # in seconds
                print(f"TIME REMAINING: {str(datetime.timedelta(seconds=remaining))}", flush=True)
 