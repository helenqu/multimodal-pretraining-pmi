import pandas as pd
import numpy as np
import argparse
from collections import Counter
from tqdm import tqdm
from pathlib import Path
import pdb
from collections import defaultdict

WORDS_TO_KEEP = 20000

def main(args, outpath):
    print('loading data...')
    input_df = pd.read_parquet(args.input)

    all_words = np.concatenate(input_df.TEXT_CLEANED.values)
    unique_words = np.unique(all_words)
    print(f"all words: {len(all_words)}")
    print(f"num unique words: {len(unique_words)}")

    freqs = Counter(all_words)
    freqs_df = pd.DataFrame(freqs.items(), columns=['word', 'frequency'])
    unique_words = freqs_df.sort_values('frequency', ascending=False).head(WORDS_TO_KEEP).word.values

    pairs_df = defaultdict(list)

    for i, row in enumerate(tqdm(input_df.itertuples(), total=len(input_df))):
        cleaned = row.TEXT_CLEANED[pd.Series(row.TEXT_CLEANED).isin(unique_words)]
        if i % 100_000 == 0:
            print(f"current num pairs: {len(pairs_df)}")

        for word_idx, word1 in enumerate(cleaned):
            for word2 in cleaned[word_idx+1:]:
                if word1 == word2:
                    continue
                pair = " ".join(sorted([word1, word2]))
                if pair not in pairs_df:
                    pairs_df[pair] = {'sample_ids': {int(row.SAMPLE_ID)}, 'frequency': 1}
                elif int(row.SAMPLE_ID) not in pairs_df[pair]['sample_ids']:
                    pairs_df[pair]['sample_ids'].add(int(row.SAMPLE_ID))
                    pairs_df[pair]['frequency'] += 1

    flat_list = []
    for key, sub_dict in pairs_df.items():
        # Create a new dictionary for each row with the key of the outer dictionary as one column
        row_dict = {'word_pair': key}
        # Add all the entries from the inner dictionary to the row dictionary
        row_dict.update(sub_dict)
        flat_list.append(row_dict)
    pairs_df = pd.DataFrame(flat_list)
    pairs_df = pd.DataFrame(pairs_df)
    pairs_df.to_parquet(outpath)

    all_pairs = [" ".join(sorted([word1, word2])) for word1 in unique_words for word2 in unique_words if word1 != word2]
    all_pairs = np.unique(all_pairs)
    with open(outpath.with_suffix(".all_pairs.txt"), "w") as f:
        f.write("\n".join(all_pairs))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='tokenize/clean laion captions, then build mapping b/t words and captions that contain each word')
    parser.add_argument('--input', type=str, help='path to cleaned laion chunk')
    args = parser.parse_args()

    outname = Path(args.input).name.split("-")[:2] + ["pair_freqs.snappy.parquet"]
    outpath = Path(args.input).parent / "-".join(outname)
    if outpath.exists():
        print(f"{outpath} already exists, skipping")
        exit()

    main(args, outpath)
