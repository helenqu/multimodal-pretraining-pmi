import numpy as np
import pandas as pd
import argparse
from collections import Counter
from tqdm import tqdm
from pathlib import Path
import gzip
import pyarrow.parquet as pq
import pdb

#TODO: reorder pairs so the imagenet word is second, or store the words in separate columns

class FrequencyFilter:
    def __init__(self, freq_threshold, input_path):
        self.freq_threshold = freq_threshold

        cleaned_path = Path(input_path).name.split('-')[:2] + ['cleaned.snappy.parquet']
        cleaned_path = '-'.join(cleaned_path)
        cleaned = pd.read_parquet(Path(input_path).parent / cleaned_path)

        all_words = np.concatenate(cleaned.TEXT_CLEANED.values)
        unique_words = np.unique(all_words)
        print(f"all words: {len(all_words)}")
        print(f"num unique words: {len(unique_words)}", flush=True)

        freqs = Counter(all_words)
        freqs_df = pd.DataFrame(freqs.items(), columns=['word', 'frequency'])
        freqs_df = freqs_df[freqs_df.frequency > freq_threshold]
        self.allowed_words = set(freqs_df.word.values)

    def filter(self, pair):
        word1, word2 = pair.split(' ')
        return word1 in self.allowed_words and word2 in self.allowed_words

class ImageNetFilter:
    def __init__(self):
        from open_clip import IMAGENET_CLASSNAMES
        import re

        imagenet_words = [re.split(r'\W+', x) for x in IMAGENET_CLASSNAMES]
        imagenet_words = [item.lower() for sublist in imagenet_words for item in sublist if len(item) > 1] # remove empty + single-char strings
        print(f"imagenet words: {imagenet_words[:10]}")
        self.imagenet_words = set(imagenet_words)

    def filter(self, pair):
        word1, word2 = pair.split(' ')
        return (word1 in self.imagenet_words) ^ (word2 in self.imagenet_words) # xor

def read_parquet(input_path):
    parquet_file = pq.ParquetFile(input_path)
    num_row_groups = parquet_file.num_row_groups
    print(f"num row groups: {num_row_groups}")

    for rg in range(num_row_groups):
        df = parquet_file.read_row_group(rg).to_pandas()
        print(f"num rows: {len(df)}")
        for i, row in tqdm(df.iterrows(), total=len(df)):
            yield row.word_pair, int(row.frequency)

def read_gzip(input_path):
    # only freq0 pairs are saved as gzip files for now, so freq should be 0/None
    with gzip.open(input_path, 'rt') as f:
        for line in tqdm(f):
            line = line.strip().split(',') # word1 word2,freq
            if len(line) == 1:
                yield line[0], 0
            else:
                yield line[0], int(line[1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tokenize/clean laion captions, then build mapping b/t words and captions that contain each word')
    parser.add_argument('--input', type=str, help='input file', required=True)
    parser.add_argument('--imagenet', action='store_true', help='filter by imagenet words')
    parser.add_argument('--freq_threshold', type=int, help='filter by frequency')
    args = parser.parse_args()

    filters = []
    if args.freq_threshold is not None:
        print(f'Filtering by frequency threshold: {args.freq_threshold}')
        filters.append(FrequencyFilter(args.freq_threshold, args.input))
    if args.imagenet:
        print('Filtering by imagenet words')
        filters.append(ImageNetFilter())

    if len(filters) == 0:
        raise ValueError('Must specify at least one filter type')

    pairs_to_keep = []
    pairs_generator = read_parquet(args.input) if args.input.endswith('.parquet') else read_gzip(args.input)

    j = 0
    for i, (pair, freq) in enumerate(pairs_generator):
        if i % 1_000_000 == 0:
            print(f"current num pairs: {len(pairs_to_keep)}, line: {i}", flush=True)
            j = 0
        if all([filter_cls.filter(pair) for filter_cls in filters]):
            pairs_to_keep.append([pair, freq])
        # DEBUGGING
        if j < 10:
            print(f'DEBUGGING, line {i}')
            print(pair, freq)
            for filter_cls in filters:
                print(filter_cls, filter_cls.filter(pair))
            j += 1
    outname = [Path(args.input).stem.split('.')[0], 'imagenet1k' if args.imagenet else '', f"freq_{args.freq_threshold}" if args.freq_threshold is not None else '', 'filtered']
    outname = '-'.join(outname)
    outpath = Path(args.input).parent / outname

    # Save filtered data
    if pairs_to_keep[0][1] > 0: # not freq0 pairs
        outpath = outpath.with_suffix('.csv')
        output_df = pd.DataFrame(pairs_to_keep, columns=['word_pair', 'frequency'])
        output_df.to_csv(outpath, index=False)
    else:
        outpath = outpath.with_suffix('.txt')
        with open(outpath, 'wt') as f:
            f.write('\n'.join([x[0] for x in pairs_to_keep]))
    print(f"Saved filtered pairs to {outpath}")
