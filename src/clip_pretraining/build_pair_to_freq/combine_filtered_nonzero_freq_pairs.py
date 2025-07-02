from pathlib import Path
import pandas as pd
import gzip
from tqdm import tqdm
import numpy as np

DATA_DIR = Path('data/laion400m-meta/cleaned')
DATA_FILES = list(DATA_DIR.glob('*pair_freqs-imagenet1k--filtered.csv'))

print(f'Found {len(DATA_FILES)} files', flush=True)
all_pairs = None
for i, data_file in enumerate(tqdm(DATA_FILES)):
    df = pd.read_csv(data_file)
    print(f'pairs in {data_file}: {len(df)}', flush=True)
    if all_pairs is None:
        all_pairs = df
    else:
        merged = pd.merge(all_pairs, df, on='word_pair', how='outer', suffixes=('_df1', '_df2'))
        merged.fillna(0, inplace=True)
        merged['frequency'] = merged['frequency_df1'] + merged['frequency_df2']
        all_pairs = merged[['word_pair', 'frequency']]
        print(f'all pairs: {len(all_pairs)}', flush=True)

outpath = DATA_DIR / 'all-nonzero-freq-pairs-imagenet1k--filtered.csv'
all_pairs.to_csv(outpath, index=False, compression='gzip')
print(f'Saved to {outpath}', flush=True)
