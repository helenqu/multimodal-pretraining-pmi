import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import random
from itertools import islice
from pathlib import Path
import datetime
import time
import pdb

from open_clip import IMAGENET_CLASSNAMES
import torch
from diffusers import FluxPipeline
from accelerate import PartialState
from accelerate.utils import gather_object
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

def distprint(content, state):
    if state.is_main_process:
        print(content)

def get_freq0_words():
    freq_threshold = 500
    freq0_pairs = pd.read_csv("data/pair_to_imagenet_label/freq0_pairs.csv")
    freq0_pairs['other_word'] = [x.split(",")[0] for x in freq0_pairs['word_pair_final']]

    nonzero_freq_pairs = pd.read_csv("data/pair_to_imagenet_label/pairs.csv")
    nonzero_freq_pairs['other_word'] = [x.split(",")[0] for x in nonzero_freq_pairs['word_pair_final']]

    freq0_other_words = []
    for word in tqdm(pd.unique(freq0_pairs['other_word']), desc="finding freq0 words that cover the most imagenet classes..."):
        x = nonzero_freq_pairs[nonzero_freq_pairs['other_word'] == word]['assigned_label_int']
        if pd.unique(x).shape[0] > freq_threshold:
            freq0_other_words.append(word)
    print(f"found {len(freq0_other_words)} freq0 words with > {freq_threshold} imagenet classes")

    return [x for x in freq0_other_words if x not in IMAGENET_CLASSNAMES]

def get_nonzero_freq_words():
    nonzero_freq_pairs = pd.read_csv("data/pair_to_imagenet_label/pairs.csv")
    nonzero_freq_pairs['other_word'] = [x.split(",")[0] for x in nonzero_freq_pairs['word_pair_final']]
    return [x for x in nonzero_freq_pairs['other_word'] if x not in IMAGENET_CLASSNAMES]

# pick n classes that cover the most imagenet classes
def get_accessory_words(n_words, freq0_accessory_words_path=None, nonzero_freq_accessory_words_path=None):
    if freq0_accessory_words_path and (freq0_accessory_words_path).exists():
        with open(freq0_accessory_words_path, "r") as f:
            freq0_words = f.read().splitlines()
        print(f"read {len(freq0_words)} freq0 words from {freq0_accessory_words_path}")
    else:
        print("no freq0 accessory words found, calculating...")
        freq0_words = random.sample(list(get_freq0_words()), n_words)
        print(f"found {len(freq0_words)} freq0 words")

        accessory_words_path = "data/accessory_images/accessory_words_freq0.txt" if not freq0_accessory_words_path else freq0_accessory_words_path
        with open(accessory_words_path, "w") as f:
            f.write("\n".join(freq0_words))

    if nonzero_freq_accessory_words_path and (nonzero_freq_accessory_words_path).exists():
        with open(nonzero_freq_accessory_words_path, "r") as f:
            nonzero_freq_words = f.read().splitlines()
        print(f"read {len(nonzero_freq_words)} nonzero freq words from {nonzero_freq_accessory_words_path}")
    else:
        print("no nonzero freq accessory words found, calculating...")
        nonzero_freq_words = get_nonzero_freq_words()
        print(f"found {len(nonzero_freq_words)} nonzero freq words")

        accessory_words_path = "data/accessory_images/accessory_words_nonzero_freq.txt" if not nonzero_freq_accessory_words_path else nonzero_freq_accessory_words_path
        with open(accessory_words_path, "w") as f:
            f.write("\n".join(nonzero_freq_words))

    return pd.unique(np.concatenate([freq0_words, nonzero_freq_words]))

def get_model():
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    return pipe

def get_segmentation_model():
    sam_checkpoint = "data/sam_vit_h_4b8939.pth" # downloaded SAM checkpoint
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    return sam

def get_batched_data(words, batch_size=32):
    template = "a {} in the center of a white background"
    data = [template.format(word) for word in words]
    batches = [
        {
            "prompts": data[i:i+batch_size],
            "words": words[i:i+batch_size]
        }
        for i in range(0, len(data), batch_size)
    ]
    return batches

def generate_accessory_images(outdir, n_freq0_images, distributed_state, batch_size=64, overwrite=False):
    words = get_accessory_words(n_freq0_images, freq0_accessory_words_path=Path(outdir) / "accessory_words_freq0.txt")
    if distributed_state.is_main_process:
        print(f"found {len(words)} accessory words")
        if not overwrite and len(list(Path(outdir).glob("*.png"))) >= len(words):
            print("accessory images already exist, skipping generation")
            return
        if overwrite and Path(outdir).exists():
            print("deleting existing accessory images")
            for path in Path(outdir).glob("*.png"):
                path.unlink()
            for path in Path(outdir).glob("*.npy"):
                path.unlink()
        print("generating accessory images")

    pipe = get_model()
    existing_accessory_images = [x.stem for x in Path(outdir).glob("*.png")]
    words = [x for x in words if x not in existing_accessory_images]
    print(f"generating {len(words)} accessory images")

    batched_data = get_batched_data(words, batch_size=batch_size)

    with distributed_state.split_between_processes(batched_data) as dist_batches:
        if distributed_state.is_main_process:
            pbar = tqdm(total=len(dist_batches), desc="Dataset progress")
            print(f"num distributed batches: {len(dist_batches)}")

        pipe.to(distributed_state.device)
        for i, batch in enumerate(dist_batches):
            distprint(f"generating images", distributed_state)
            images = pipe(
                batch['prompts'],
                height=512,
                width=512,
                guidance_scale=5.0,
                num_inference_steps=28,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(0),
            ).images
            distprint(f"generated {len(images)} images", distributed_state)

            distprint(f"saving images", distributed_state)
            for j, image in enumerate(images):
                image.save(outdir / f"{batch['words'][j]}.png")
            
            if distributed_state.is_main_process:
                pbar.update(1)
                
                if i % 10 == 0:
                    rate = pbar.format_dict["rate"]
                    remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0 # in seconds
                    print(f"TIME REMAINING: {str(datetime.timedelta(seconds=remaining))}", flush=True)

    pipe.to("cpu")

def sam_prepare_image(image, transform, device=None):
    image = transform.apply_image(image)
    image = torch.as_tensor(image) 
    if device is not None:
        image = image.to(device)
    return image.permute(2, 0, 1).contiguous()

if __name__ == "__main__":
    outdir = Path("data/accessory_images")
    if not outdir.exists():
        try:
            outdir.mkdir(parents=True)
        except FileExistsError as e: # race condition
            print(f"error creating outdir {outdir}: {e}, trying in 5 seconds...")
            time.sleep(5)
            if not outdir.exists():
                outdir.mkdir(parents=True)

    distributed_state = PartialState()

    n_freq0_images = 5000
    generate_accessory_images(outdir, n_freq0_images, distributed_state, overwrite=False)
    time.sleep(5)

    sam = get_segmentation_model()

    # Read images in batches from outdir
    batch_size = 8
    all_image_paths = list(outdir.glob("*.png"))
    print(f"found {len(all_image_paths)} images")
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    
    image_batches = []
    image_size = np.array(Image.open(all_image_paths[0])).shape[:2]
    bounding_box = [128, 128, 384, 384] #middle square for 512x512 images
    for i in range(0, len(all_image_paths), batch_size):
        batch_paths = all_image_paths[i:i + batch_size]
        sam_batch = [
            {
                'image': sam_prepare_image(np.array(Image.open(path)), resize_transform),
                'point_coords': resize_transform.apply_coords_torch(torch.tensor([[[0, 0], [image_size[0]-1, image_size[1]-1], [1, image_size[1]-1], [image_size[0]-1, 1]]]), image_size),
                'original_size': image_size,
                'image_paths': path,
                'point_labels': torch.tensor([[1, 1, 1, 1]])
            }
            for path in batch_paths
        ]
        if i == 0:
            print(sam_batch[0]['image'].shape)
        image_batches.append(sam_batch)
    print(f'processing {len(image_batches)} batches')

    quality_scores_per_process = []
    image_paths_per_process = []
    with distributed_state.split_between_processes(image_batches) as dist_batches:
        for i, sam_batch in enumerate(dist_batches):
            distprint(f"running SAM on {len(sam_batch)} images", distributed_state)
            image_paths = [image['image_paths'] for image in sam_batch]
            
            sam.to(distributed_state.device)
            sam_batch = [
                {k: v.to(distributed_state.device) if k != 'original_size' else v # don't move original_size to device
                 for k, v in image.items()
                 if k != 'image_paths'} # remove image_paths from batch
                for image in sam_batch
            ]
            sam_output = sam(sam_batch, multimask_output=False)
            if len(sam_output) < 8:
                print(f"only {len(sam_output)} / 8 masks found for batch {i}")
            
            # Save masks
            for j, mask in enumerate(sam_output):
                np.save(outdir / f"{image_paths[j].stem}_bg_mask.npy", mask['masks'].squeeze().cpu().numpy())
            
            # Save IOU predictions to dataframe
            quality_scores_per_process.extend([mask['iou_predictions'].squeeze().cpu().numpy() for mask in sam_output])
            image_paths_per_process.extend(image_paths)
            
            if i % 10 == 0:
                print(f"done batch {i}")
    
    # Gather quality scores from all processes and save to dataframe
    all_quality_scores = gather_object(quality_scores_per_process)
    all_image_paths = gather_object(image_paths_per_process)
    if distributed_state.is_main_process:
        quality_scores_df = pd.DataFrame({'iou_prediction': all_quality_scores, 'image_path': all_image_paths})
        quality_scores_df.to_csv(outdir / 'quality_scores.csv', index=False)
        print(f"Saved quality scores for {len(quality_scores_df)} masks")