from pathlib import Path
import pandas as pd
import gzip
from tqdm import tqdm
import numpy as np

DATA_DIR = Path('data/laion400m-meta/cleaned')
DATA_FILES = list(DATA_DIR.glob('*freq0-pairs-imagenet1k--filtered.gz'))
NONZERO_FREQ_FILES = list(DATA_DIR.glob('*pair_freqs.snappy.parquet'))

print(f'Found {len(DATA_FILES)} freq0 files and {len(NONZERO_FREQ_FILES)} nonzero freq files', flush=True)
all_freq0_pairs = []
for data_file in DATA_FILES:
    print(data_file)
    with open(data_file, 'r') as f:
        all_freq0_pairs += [line.strip() for line in f]

print(f'all pairs: {len(all_freq0_pairs)}', flush=True)
for freq_file in tqdm(NONZERO_FREQ_FILES):
    df = pd.read_parquet(freq_file)
    nonzero_freq_pairs = set(df.word_pair.values)
    all_freq0_pairs = [pair for pair in all_freq0_pairs if pair not in nonzero_freq_pairs]
    print(f'filtered pairs: {len(all_freq0_pairs)}', flush=True)
all_freq0_pairs = pd.unique(all_freq0_pairs)

with gzip.open(DATA_DIR / 'all-freq0-pairs-imagenet1k-filtered.gz', 'wt') as f:
    f.write('\n'.join(all_freq0_pairs))
print("done!")
