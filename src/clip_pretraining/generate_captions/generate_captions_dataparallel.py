# this script is allocated a whole GPU node and uses data parallelism to split the data across GPUs
# take chunk of pairs df, further split into n_gpus chunks

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import PartialState
from accelerate.utils import gather_object

import argparse
import pandas as pd
from open_clip import IMAGENET_CLASSNAMES
import pdb
from tqdm import tqdm
from pathlib import Path
import datetime
import re

from .llm_filtering import filter_captions

USER_PROMPT = lambda pair: f"Please write a single sentence that could describe an image that contains the words '{pair[0]}' and '{pair[1]}'. Make sure both {pair[0]} and {pair[1]} are the focus of the image."
PROMPT = lambda pair: [{"role": "system", "content": "You are a helpful assistant that creates prompts for image generation."}, {"role": "user", "content": USER_PROMPT(pair)}]

SINGLE_WORD_IMAGENET_CLASSNAMES = [re.sub(r'[^a-zA-Z]+', '', x.split()[-1]) for x in IMAGENET_CLASSNAMES]

def get_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    return tokenizer

def tokenize(batch, tokenizer):
    templated = [tokenizer.apply_chat_template(prompt,
                                              add_generation_prompt=True,
                                              tokenize=False,
                                              ) for prompt in batch]
    print(templated[0])
    tokenized = tokenizer(
        templated,
        padding=True,
        pad_to_multiple_of=8, # for fp16 on tensor core gpus
        return_tensors="pt"
    )
    return {
        'input_ids': tokenized.input_ids,
        'attention_mask': tokenized.attention_mask
    }


def get_data(args, tokenizer):
    num_gpus = torch.cuda.device_count()
    print(f"found {num_gpus} gpus")
    input_data = pd.read_csv(args.pairs_df)
    orig_len = len(input_data)
    if args.num_to_generate and len(input_data) > args.num_to_generate:
        input_data = input_data.sample(n=args.num_to_generate, random_state=1)
        print(f"sampled {len(input_data)} examples")
    print(f"generating captions for {len(input_data)} examples")

    all_prompts = []
    batch = {"ids": [], "prompts": []}
    for i, (_, row) in enumerate(input_data.iterrows()):
        if i % args.batch_size == 0 and i > 0:
            all_prompts.append(batch)
            batch = {"ids": [], "prompts": []}
        pair = row.word_pair_final.split(',')
        if pair[1] == "orange": # hacky fix for orange showing up as a color
            pair[1] = "orange (fruit)"
        batch['ids'].append(row.word_pair_id)
        batch['prompts'].append(PROMPT(pair))

    if len(batch) > 0:
        all_prompts.append(batch)

    tokenized_prompts = [
        [batch['ids'], tokenize(batch['prompts'], tokenizer)] for batch in all_prompts
    ]

    return tokenized_prompts, input_data

def filter_captions(row):
    # remove captions with imagenet class names, excluding the assigned label
    classnames_to_check = SINGLE_WORD_IMAGENET_CLASSNAMES[:row['assigned_label_int']] + SINGLE_WORD_IMAGENET_CLASSNAMES[row['assigned_label_int']+1:]
    for classname in classnames_to_check:
        if classname.lower() in row['caption'].lower().split():
            print(f"filtered out by {classname}: {row['word_pair']}: {row['caption']}")
            return False

    return True

def get_model(model_id):
    model = AutoModelForCausalLM.from_pretrained(model_id).cpu()
    model.eval()
    model.resize_token_embeddings(model.config.vocab_size + 1)
    return model

def distprint(content, state):
    if state.is_main_process:
        print(content)

distributed_state = PartialState()

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int)
parser.add_argument("--pairs_df", type=str, help="output of pair_to_imagenet_label (incl LLM filtering)")
parser.add_argument("--num_to_generate", type=int, help="number of captions to generate")
parser.add_argument("--output_path", type=str, help="output path")
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument("--min_p", type=float, default=0.05)
args = parser.parse_args()
distprint(vars(args), distributed_state)


model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = get_tokenizer(model_id)
tokenized_prompts, input_data = get_data(args, tokenizer)

model = get_model(model_id)

distprint(f"loaded model and dataset with {len(tokenized_prompts)} batches", distributed_state)

ids_per_process = []
outputs_per_process = []

with distributed_state.split_between_processes(tokenized_prompts) as batched_prompts:
    if distributed_state.is_main_process:
        pbar = tqdm(total=len(batched_prompts), desc="Processing")


    # each batch is a list: [pairs, tokenized_prompts]
    for i, (ids, batch) in enumerate(batched_prompts):
        batch = {k: v.to(distributed_state.device) for k, v in batch.items()}

        with torch.inference_mode():
            model.to(distributed_state.device)
            outputs = model.generate(
                    **batch,
                    do_sample=True,
                    temperature=args.temperature,
                    max_new_tokens=50,
                    pad_token_id=tokenizer.pad_token_id,
                    min_p=args.min_p
            )
            # select only generated tokens
            outputs = outputs[:, batch['input_ids'].shape[1]:].cpu()
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        outputs_per_process.extend(decoded)
        ids_per_process.extend(ids)

        if distributed_state.is_main_process:
            pbar.update(1)
            if i % 10 == 0:
                print(decoded[0])
                rate = pbar.format_dict["rate"]
                remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0 # in seconds
                print(f"TIME REMAINING: {str(datetime.timedelta(seconds=remaining))}")

num_examples = len(input_data)
all_outputs = gather_object(outputs_per_process)
print(f"gathered outputs: {len(all_outputs)}")
all_outputs = all_outputs[:num_examples] # remove duplicated batches
all_ids = gather_object(ids_per_process)
all_ids = all_ids[:num_examples] # remove duplicated batches

filtered = input_data.loc[input_data['word_pair_id'].isin(all_ids)]
ordered_data = filtered.set_index('word_pair_id').loc[all_ids].drop_duplicates() # reorder
all_pairs = ordered_data['word_pair_final'].tolist()
all_imgnet_ids = ordered_data['assigned_label_int'].tolist()
if 'frequency' not in ordered_data.columns:
    ordered_data['frequency'] = 0.
all_freqs = ordered_data['frequency'].tolist()

print(len(all_ids), len(all_outputs), len(all_pairs), len(all_imgnet_ids))
output_df = pd.DataFrame({
    "word_pair_id": all_ids,
    "caption": [x.split("\n")[0] for x in all_outputs],
    'word_pair': all_pairs,
    'assigned_label_int': all_imgnet_ids,
    'frequency': all_freqs
})
print(output_df.head())

outdir = Path("data/clip_pretraining_metadata/captions")
outpath = outdir / f"captions__{Path(args.pairs_df).stem}.csv" if not args.output_path else Path(args.output_path)
if outpath.exists():
    # add today's date to the filename
    outpath = outpath.with_name(f"{outpath.stem}__{datetime.date.today().strftime('%Y-%m-%d')}.csv")
else:
    outpath.parent.mkdir(parents=True, exist_ok=True)

prompt_example = tokenizer.apply_chat_template(PROMPT(('gong', 'copycat')), add_generation_prompt=True, tokenize=False)
prompt_example = "\n".join(["#" + x for x in prompt_example.split("\n")])
with open(outpath, 'w') as f:
    f.write("#" + prompt_example + "\n")
    f.write(output_df.to_csv(index=False))

print(f"Captions written to {outpath}")