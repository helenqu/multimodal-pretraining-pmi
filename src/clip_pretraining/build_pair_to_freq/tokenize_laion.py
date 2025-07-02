import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
import string
import re
import argparse
from collections import Counter
from tqdm import tqdm
from pathlib import Path
import pdb

FREQ_THRESHOLD = 300

parser = argparse.ArgumentParser(description='tokenize/clean laion captions, then build mapping b/t words and captions that contain each word')
parser.add_argument('--input', type=str, help='path to laion chunk')
args = parser.parse_args()

outname = Path(args.input).name.split("-")[:2] + ["cleaned.snappy.parquet"]
outpath = Path(args.input).parent / "-".join(outname)
if outpath.exists():
    print(f"{outpath} already exists, skipping")
    exit()

# tokenizing and cleaning

print('loading data...')
input_df = pd.read_parquet(args.input)
input_df = input_df[(~pd.isna(input_df.TEXT)) & (~pd.isna(input_df.SAMPLE_ID))].reset_index()
print(input_df.TEXT.isna().sum(), input_df.SAMPLE_ID.isna().sum())

print('tokenizing...')
tokenized = [re.sub('[\W_]+', ' ', caption).lower().split() for caption in tqdm(input_df.TEXT.values)]

stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()

print('cleaning...')
cleaned = [[lemmatizer.lemmatize(x) for x in tokenized_caption if x not in stopwords and len(x) > 1 and not x.isnumeric()] for tokenized_caption in tqdm(tokenized)]
print('assign into input_df')
input_df['TEXT_CLEANED'] = cleaned

print(f"saving to {outpath}")
input_df[['SAMPLE_ID', 'TEXT_CLEANED']].to_parquet(outpath)
print("done!")
