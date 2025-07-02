import argparse
import pandas as pd
import random
from open_clip import IMAGENET_CLASSNAMES
import re
from tqdm import tqdm
import gzip
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from pathlib import Path
import time
import pdb

from .llm_filtering import filter_visualizable_words

parser = argparse.ArgumentParser()
parser.add_argument("--pairs_file", type=str, default=None, help="e.g. data/laion400m-meta/all-nonzero-freq-pairs-imagenet1k--filtered.csv.gz")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for LLM filtering")
parser.add_argument("--no_llm_filtering", action='store_true', default=True, help="do not apply LLM filtering for visualizability")
parser.add_argument("--outdir", type=str, default="data/pair_to_imagenet_label", help="output directory")
args = parser.parse_args()

def read_csv(input_file):
    df = pd.read_csv(input_file)
    print(f"Read {len(df)} rows from {input_file}")
    return df

def read_txt(input_file):
    open_func = gzip.open if input_file.endswith('.gz') else open
    with open_func(input_file) as f:
        data = [x.decode('utf-8').strip() for x in f.readlines()]
    df = {'word_pair': data,
          'frequency': [0] * len(data)}
    return pd.DataFrame(df)

def split_pair(row):
    word1, word2 = row.word_pair.split()
    if word1 in imagenet_categories.keys():
        return word2, word1
    elif word2 in imagenet_categories.keys():
        return word1, word2
    else:
        return None, None

# filter other words
def other_words_filter(other_word):
    # remove words with digits or words not in wordnet synsets (proxy for non-words)
    if re.search(r'\d', other_word) or not bool(wn.synsets(other_word)):
        return False
    # remove non-nouns or non-adjectives
    word_pos_tag = pos_tag([other_word])[0][1]
    is_noun_or_adj = word_pos_tag.startswith('N') or word_pos_tag.startswith('J')
    if not is_noun_or_adj:
        return False
    if 'photo' in other_word or 'image' in other_word: # image generator won't do anything with these
        return False
    return True

def choose_imagenet_label(word, categories):
    labels_for_pair = []
    for category, indices in categories.items():
        if word == category:
            labels_for_pair.extend(indices)

    if len(labels_for_pair) == 0:
        print(f"No imagenet label found for {word}")
        return -1, 'none'
    assigned_int = random.sample(labels_for_pair, 1)[0]
    assigned = IMAGENET_CLASSNAMES[assigned_int]

    return assigned_int, assigned

outdir = Path(args.outdir)
outpath = outdir / Path(args.pairs_file).name
print(f"outputting to {outpath}")

# split imagenet labels into categories based on last word, match by category
imagenet_categories = {}
for i, classname in enumerate(IMAGENET_CLASSNAMES):
    classname = classname.lower()
    last_word = classname.split()[-1]
    if last_word in imagenet_categories:
        last_word = re.sub(r'[^a-zA-Z]+', '', last_word) # remove non-alphabetic characters
        imagenet_categories[last_word].append(i)
    else:
        imagenet_categories[last_word] = [i]

pairs_df = read_csv(args.pairs_file) if '.csv' in args.pairs_file else read_txt(args.pairs_file)
print(f"Read {len(pairs_df)} rows from {args.pairs_file}")
pairs_df['other_word'], pairs_df['imagenet_word'] = zip(*pairs_df.apply(split_pair, axis=1))
print(f"before dropping na {len(pairs_df)}")
pairs_df.dropna(subset=['other_word'], inplace=True)
print(f"after dropping na {len(pairs_df)}")

# filter other words
other_words = pairs_df['other_word'].unique().tolist()
print(f"num unique other words: {len(other_words)}")
other_words = [x for x in other_words if other_words_filter(x)]
print(f"num other words after basic filtering: {len(other_words)}")

if not args.no_llm_filtering:
    # use llm filtering
    other_words = filter_visualizable_words(other_words, outdir, args.batch_size)
    print(f"num other words after LLM filtering: {len(other_words)}")
print(other_words[:10])
pairs_df = pairs_df[pairs_df['other_word'].isin(other_words)]

# assign imagenet label
label_mapping = defaultdict(list)
both = []
not_word = []
non_nouns_adjs = []
start = time.time()
for i, row in enumerate(pairs_df.itertuples()):
    if i % 500_000 == 0:
        print(f"Processed {i} pairs in {time.time() - start:.2f} seconds", flush=True)

    assigned_int, assigned = choose_imagenet_label(row.imagenet_word, imagenet_categories)
    if assigned_int < 0:
        continue
    word_pair_final = f"{row.other_word},{assigned}"

    label_mapping['word_pair'].append(row.word_pair)
    label_mapping['assigned_label_text'].append(assigned)
    label_mapping['assigned_label_int'].append(assigned_int)
    label_mapping['word_pair_final'].append(word_pair_final)
    if row.frequency > 0:
        label_mapping['frequency'].append(row.frequency)

print(f"final label mapping num: {len(label_mapping['word_pair'])}", flush=True)

label_mapping_df = pd.DataFrame(label_mapping)
label_mapping_df = label_mapping_df.dropna().drop_duplicates(subset=['word_pair'], keep='first')
label_mapping_df = label_mapping_df.reset_index(drop=False).rename(columns={'index': 'word_pair_id'})
print(label_mapping_df.head())

label_mapping_df.to_csv(outpath, index=False)
print(f"Saved label mapping to {outpath}")