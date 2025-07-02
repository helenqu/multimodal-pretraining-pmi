import argparse
import json
import pandas as pd
import re
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
from pathlib import Path
import pdb
import multiprocessing as mp
import time
from collections import defaultdict
import duckdb

def search_in_file(args):
    file_path, search_values = args
    """Worker function: Opens a file once, runs all searches, and sends results."""
    conn = duckdb.connect(database=":memory:")
    
    # Load Parquet file into DuckDB
    conn.execute(f"CREATE TABLE data AS SELECT * FROM parquet_scan('{file_path}')")
    print(f"loaded {file_path}", flush=True)
    
    # Use IN clause for efficient batch searching
    query = f"SELECT word_pair, frequency FROM data WHERE word_pair IN ({','.join(['?'] * len(search_values))})"
    result = conn.execute(query, search_values).fetch_df()
    print("finished query", flush=True)
    
    # result_queue.put(result)
    conn.close()
    return result

# for generating word pairs before searching for freqs
def get_word_pairs(questions_file, annotations_file=None, preds=None):
    print("pair freq csv not found, computing for", Path(questions_file).parent.name)

    assert not (annotations_file == None and preds == None), "must provide either annotations or predictions"

    if questions_file.endswith(".json"):
        questions = json.load(open(questions_file, 'r'))["questions"]
        questions = pd.DataFrame(questions)
    elif questions_file.endswith(".jsonl"):
        questions = [json.loads(line) for line in open(questions_file, 'r')]
        questions = pd.DataFrame(questions)
    questions.rename(columns={"text": "question"}, inplace=True)
    questions['question'] = ['\n'.join(x.split('\n')[:-1]) for x in questions['question']] # remove prompt

    if annotations_file is not None:
        print("getting gold answers")
        if annotations_file.endswith(".json"):
            answers_wrapper = json.load(open(annotations_file, 'r'))
            if "annotations" in answers_wrapper:
                answers = answers_wrapper["annotations"]
            else:
                answers = answers_wrapper["data"]
            
            for entry in answers:
                if isinstance(entry["answers"], dict):
                    entry["answers"] = [x["answer"] for x in entry["answers"]]
                entry["gold_answer"] = max(entry["answers"], key=entry["answers"].count)

            answers = pd.DataFrame(answers)
            # qa = pd.merge(questions, answers, on="question_id", how="inner")
        elif annotations_file.endswith(".jsonl"):
            answers = [x['answers'] for x in json.load(open(annotations_file, 'r'))['data']]
            # qa = pd.DataFrame({"question": questions, "answers": answers})
            answers = pd.DataFrame({"answers": answers})
            answers["gold_answer"] = answers["answers"].apply(lambda x: max(x, key=x.count))
    
    if preds is not None:
        print("getting predictions")
        answers = pd.read_csv(preds)
        answers.rename(columns={"answer": "pred_answer"}, inplace=True)
        answers['pred_answer'] = answers['pred_answer'].apply(lambda x: str(x))
    
    if "textvqa" in questions_file:
        answers.drop(columns=["question"], inplace=True)
        qa = pd.concat([questions, answers], axis=1)
        qa["question"] = qa["question"].str.replace("Reference OCR token: ", "")
    else:
        qa = pd.merge(questions, answers, on="question_id", how="inner")

    if preds is not None:
        qa["qa_str"] = qa["question"].str.replace("?", "") + " " + qa["pred_answer"]
    else:
        qa["qa_str"] = qa["question"].str.replace("?", "") + " " + qa["gold_answer"]
    tokenized = [re.sub('[\W_]+', ' ', caption).lower().split() for caption in tqdm(qa.qa_str.values)]

    stopwords = [x for x in nltk.corpus.stopwords.words('english') if x not in ["yes", "no"]]
    lemmatizer = WordNetLemmatizer()
    print('cleaning...')
    cleaned_tokenized = [[lemmatizer.lemmatize(word) for word in qa_entry if word not in stopwords and len(word) > 1 and not word.isnumeric()] for qa_entry in tqdm(tokenized)]

    pairs = []
    for qa_entry in tqdm(cleaned_tokenized):
        pairs_for_entry = []
        for word_idx, word1 in enumerate(qa_entry):
            for word2 in qa_entry[word_idx+1:]:
                if word1 == word2:
                    continue
                pair = " ".join(sorted([word1, word2]))
                pairs_for_entry.append(pair)
        pairs.append(pd.unique(pairs_for_entry))
    qa["pairs"] = pairs

    if "image_id_x" in qa.columns:
        qa.drop(columns=["image_id_x"], inplace=True)
        qa.rename(columns={"image_id_y": "image_id"}, inplace=True)
    
    outfile = Path(questions_file).parent / f"llava_qa_word_pairs_{'pred' if preds is not None else 'gold'}.csv"
    print(f"saving {outfile}")
    qa.to_csv(outfile, index=False)
    return qa


parser = argparse.ArgumentParser()
parser.add_argument('--questions', type=str, help='path to questions file')
parser.add_argument('--annotations', type=str, help='path to answers file')
parser.add_argument('--predictions', type=str, help='path to predictions file')
parser.add_argument('--out_file', type=str, help='path to predictions file', default=None)
parser.add_argument('--read_from', type=str, help='path to word pairs file', default=None)
parser.add_argument('--compute_pairs', action='store_true', help='compute pairs')
parser.add_argument('--process_freqs', action='store_true', help='process freqs')
args = parser.parse_args()

if args.compute_pairs:
    print("computing pairs")
    pairs = get_word_pairs(args.questions, args.annotations, preds=args.predictions)
else:
    if args.read_from is None:
        out_path = Path(args.questions).parent / f"llava_qa_word_pairs.csv"
    else:
        out_path = args.read_from
    print(f"reading pairs from {out_path}")
    pairs = pd.read_csv(out_path)
    pairs['pairs'] = pairs['pairs'].apply(lambda x: eval(x.replace(" ", ",")))
print(pairs.head())
print(f"now go edit and run scripts/get_vqa_pair_freqs.sh to get freqs!")

# after getting freqs, aggregate across files and add to pairs df
if args.process_freqs:
    print("processing freqs")
    freq_files = list((Path(args.questions).parent / "llava_format_freqs_pred").glob("*.csv"))
    all_freqs = None
    for path in freq_files:
        a = pd.read_csv(path)
        if all_freqs is None:
            all_freqs = a
        else:
            all_freqs = pd.concat([all_freqs, a]).groupby("word_pair", as_index=False).sum()
    print(all_freqs.head())

    nonzero_freq_pairs = set(all_freqs['word_pair'].values)
    laion_freqs = []
    for i, row in tqdm(pairs.iterrows(), total=len(pairs)):
        row_pairs = [x for x in row.pairs if isinstance(x, str)] # get rid of ellipsis objects
        row_pairs = [pair.replace(",", " ") for pair in row_pairs]
        laion_freqs_for_row = [all_freqs[all_freqs['word_pair'] == pair].frequency.values[0] if pair in nonzero_freq_pairs else 0 for pair in row_pairs]
        laion_freqs.append(laion_freqs_for_row)
        if i == 1:
            print(row_pairs)
            print(laion_freqs_for_row)
            print(laion_freqs)
    pairs["laion_freqs"] = laion_freqs
    pairs.to_csv(Path(args.questions).parent / "llava_qa_word_pairs_pred_with_freqs.csv", index=False)
