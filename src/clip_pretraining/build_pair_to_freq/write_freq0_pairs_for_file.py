import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import gzip

import gc
parser = argparse.ArgumentParser(description='get frequency 0 word pairs for laion chunk')
parser.add_argument('--input', type=str, help='path to cleaned laion chunk')
args = parser.parse_args()

basename = Path(args.input).name.split("-")[:2]
outname = basename + ["freq0-pairs.gz"]
outpath = Path(args.input).parent / "-".join(outname)
print(outpath, flush=True)
if outpath.exists():
    print(f"{outpath} already exists, skipping", flush=True)
    exit()

print(f"reading from {args.input}", flush=True)
input_df = pd.read_parquet(args.input)

from collections import Counter

WORDS_TO_KEEP = 20_000

all_words = np.concatenate(input_df.TEXT_CLEANED.values)
unique_words = np.unique(all_words)
print(f"all words: {len(all_words)}", flush=True)
print(f"num unique words: {len(unique_words)}", flush=True)
del input_df
gc.collect()

freqs = Counter(all_words)
freqs_df = pd.DataFrame(freqs.items(), columns=['word', 'frequency'])
unique_words = freqs_df.sort_values('frequency', ascending=False).head(WORDS_TO_KEEP).word.values
del freqs_df
gc.collect()

existing_pairs_path = Path(args.input).parent / "-".join(basename + ["pair_freqs.snappy.parquet"])
assert existing_pairs_path.exists(), f"{existing_pairs_path} does not exist"

existing_pairs_df = pd.read_parquet(existing_pairs_path)
existing_pairs = set(existing_pairs_df.word_pair.values)
del existing_pairs_df

freq_0_pairs = []

for i, word_i in enumerate(tqdm(unique_words)):
    for j, word_j in enumerate(unique_words[i+1:]):
        if i % 1_000 == 0 and j == 0:
            print(len(freq_0_pairs), flush=True)
        if word_i == word_j:
            continue
        pair = " ".join(sorted([word_i, word_j]))
        if pair not in existing_pairs:
            freq_0_pairs.append(pair)

print(f"writing {len(freq_0_pairs)} pairs to {outpath}", flush=True)
data = "\n".join(freq_0_pairs)

# Compress and save to a file
with gzip.open(outpath, 'wt', encoding='utf-8') as f:
    f.write(data)
