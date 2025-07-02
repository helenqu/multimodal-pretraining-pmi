from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import PartialState
from accelerate.utils import gather_object

import pandas as pd
from tqdm import tqdm
from pathlib import Path
import datetime
import re
from unidecode import unidecode

OUTPATH = "data/clip_pretraining/metadata"

def get_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    return tokenizer

def tokenize(batch, tokenizer):
    tokenized = tokenizer(
        batch,
        padding=True,
        pad_to_multiple_of=8, # for fp16 on tensor core gpus
        return_tensors="pt"
    )
    return {
        'input_ids': tokenized.input_ids,
        'attention_mask': tokenized.attention_mask
    }

def get_one_batch(data, i, batch_size):
    if isinstance(data, pd.DataFrame):
        return data.iloc[i:i+batch_size]
    else:
        return data[i:i+batch_size]

def batchify(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield get_one_batch(data, i, batch_size)

def prepare_data(words, batch_size, tokenizer):
    with open(Path(__file__).parent / "llm_filtering_prompt.txt", 'r') as f:
        prompt_template = f.read()

    prompt = lambda word: prompt_template.format(word)

    all_prompts = [
        [batch,
        tokenize([prompt(word) for word in batch], tokenizer)] for batch in batchify(words, batch_size)
    ]

    return all_prompts

def prepare_captions(captions, batch_size, tokenizer):
    with open(Path(__file__).parent / "filter_captions_prompt.txt", 'r') as f:
        prompt_template = f.read()

    user_prompt = lambda caption, classname, other_word: prompt_template.format(caption, classname, other_word, classname, other_word)
    prompt = lambda caption, classname, other_word: user_prompt(caption, classname, other_word)

    all_prompts = [
        [batch['word_pair_id'].tolist(),
        tokenize([
            prompt(row.caption, 
                   row.word_pair.split(",")[1], 
                   row.word_pair.split(",")[0]) for row in batch.itertuples()
            ], tokenizer)
        ] for batch in batchify(captions, batch_size)
    ]

    return all_prompts

def get_model(model_id):
    model = AutoModelForCausalLM.from_pretrained(model_id).cpu()
    model.eval()
    model.resize_token_embeddings(model.config.vocab_size + 1)
    return model

def run_one_batch(model, state, batch, tokenizer, output_parser=None):
    with torch.inference_mode():
        model.to(state.device)
        outputs = model.generate(
                **batch,
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                num_beams=3,
                length_penalty=1.2,
                max_new_tokens=250,
                pad_token_id=tokenizer.pad_token_id
        )
        # select only generated tokens
        outputs = outputs[:, batch['input_ids'].shape[1]:].cpu()
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    if output_parser:
        output = [output_parser(x) for x in decoded]
    else:
        output = decoded
    return output

def print_progress(pbar, decoded):
    print(decoded[0])
    rate = pbar.format_dict["rate"]
    remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0 # in seconds
    print(f"TIME REMAINING: {str(datetime.timedelta(seconds=remaining))}", flush=True)

def distprint(content, state):
    if state.is_main_process:
        print(content)

def filter_visualizable_words(words, outpath, batch_size=8):
    def parse_output(output):
        sentences = re.split(r'[^a-zA-Z0-9 ]+', output)
        answer_sentence = [s for s in sentences if s.strip().startswith("The answer is")]
        if len(answer_sentence) > 1:
            answer_sentence = [s for s in answer_sentence if "yes" in s or "no" in s]
        if len(answer_sentence) == 0:
            return "NO ANSWER FOUND"
        return answer_sentence[-1].split()[-1]

    distributed_state = PartialState()

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model = get_model(model_id)

    tokenizer = get_tokenizer(model_id)
    tokenized_prompts = prepare_data(words, batch_size, tokenizer)

    distprint(f"loaded model and dataset with {len(tokenized_prompts)} batches", distributed_state)

    words_per_process = []
    outputs_per_process = []

    with distributed_state.split_between_processes(tokenized_prompts) as batched_prompts:
        if distributed_state.is_main_process:
            pbar = tqdm(total=len(batched_prompts), desc="Processing")
            # DEBUGGING
            print(f"num batches: {len(batched_prompts)}")

        # each batch is a list: [pairs, tokenized_prompts]
        for i, (words, batch) in enumerate(batched_prompts):
            batch = {k: v.to(distributed_state.device) for k, v in batch.items()}

            decoded = run_one_batch(model, distributed_state, batch, tokenizer, output_parser=parse_output)

            outputs_per_process.extend(decoded)
            words_per_process.extend(words)

            if distributed_state.is_main_process:
                pbar.update(1)
                if i % 10 == 0:
                    print_progress(pbar, decoded)
    # if distributed_state.is_main_process:
    all_outputs = gather_object(outputs_per_process)
    print(f"gathered outputs: {len(all_outputs)}")
    all_words = gather_object(words_per_process)

    output_df = pd.DataFrame({
        "other_word": all_words,
        "is_visualizable": all_outputs,#[parse_output(x) for x in all_outputs],
    })
    print(output_df.head())

    outpath = Path(outpath) / "llm_filtered_words.csv"
    output_df.to_csv(outpath, index=False)
    print(f"filtering results written to {outpath}")

    return output_df[output_df['is_visualizable'] == "yes"]['other_word'].tolist()


def filter_captions(captions, batch_size=16):
    def filter_outputs(captions, outputs):
        filtered_outputs = []
        for i, (parsed_objects, output) in enumerate(outputs):
            word_pair = captions.iloc[i]['word_pair']
            other_word = word_pair.split(',')[0]
            imagenet_class = word_pair.split(',')[1]
            valid_responses = [x for x in parsed_objects if unidecode(imagenet_class).lower() in unidecode(x['rewritten_sentence']).lower()]
            if len(valid_responses) > 0:
                filtered_outputs.append([
                    valid_responses[0]['has_imagenet_class'], 
                    valid_responses[0]['rewritten_sentence'],
                    other_word.lower() in valid_responses[0]['rewritten_sentence'].lower(),
                    output
                ])
            else:
                filtered_outputs.append([None, None, None, output])
        return filtered_outputs

    def parse_output(output):
        # Regular expression to match JSON objects
        json_pattern = r'{.*?}'
        
        # Search for JSON-like structures
        matches = re.findall(json_pattern, output)
        parsed_objects = []
        for match in matches:
            try:
                # Try parsing the JSON object
                response = eval(match)
                assert 'rewritten_sentence' in response and 'has_imagenet_class' in response
                parsed_objects.append(response)
            except (AssertionError, ValueError, SyntaxError, TypeError):
                # Skip invalid JSON objects
                continue
        if len(parsed_objects) == 0 or type(parsed_objects[0]) != dict:
           return [[], output]

        return [parsed_objects, output]
    
    distributed_state = PartialState()

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model = get_model(model_id)

    tokenizer = get_tokenizer(model_id)
    tokenized_prompts = prepare_captions(captions, batch_size, tokenizer)

    distprint(f"loaded model and dataset with {len(tokenized_prompts)} batches", distributed_state)

    ids_per_process = []
    outputs_per_process = []

    with distributed_state.split_between_processes(tokenized_prompts) as batched_prompts:
        if distributed_state.is_main_process:
            pbar = tqdm(total=len(batched_prompts), desc="Filtering")
            # DEBUGGING
            print(f"num batches: {len(batched_prompts)}")

        # each batch is a list: [pairs, tokenized_prompts]
        for i, (ids, batch) in enumerate(batched_prompts):
            batch = {k: v.to(distributed_state.device) for k, v in batch.items()}

            decoded = run_one_batch(model, distributed_state, batch, tokenizer, output_parser=parse_output)

            outputs_per_process.extend(decoded)
            ids_per_process.extend(ids)

            if distributed_state.is_main_process:
                pbar.update(1)
                if i % 10 == 0:
                    print_progress(pbar, decoded)

    all_outputs = gather_object(outputs_per_process)
    print(f"gathered outputs: {len(all_outputs)}")
    all_ids = gather_object(ids_per_process)

    captions = captions.set_index('word_pair_id').reindex(all_ids).reset_index()
    all_outputs = filter_outputs(captions, all_outputs)

    output_df = pd.DataFrame({
        "word_pair_id": all_ids,
        "has_imagenet_class": [x[0] for x in all_outputs],
        "caption": [x[1] for x in all_outputs],
        "other_word_in_sentence": [x[2] for x in all_outputs],
        "raw_response": [x[3] for x in all_outputs],
    })
    captions['orig_caption'] = captions['caption']
    captions = captions.drop(columns=['caption'])
    output_df = output_df.merge(captions, on='word_pair_id', how='left')
    print(output_df.head())

    return output_df
