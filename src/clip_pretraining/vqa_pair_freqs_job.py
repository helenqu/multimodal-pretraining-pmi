import argparse
import duckdb
import pandas as pd
from pathlib import Path
import pdb

def get_freqs_for_file(search_values, file_path, output_path):
    try:
        print(f"Processing {file_path}, writing to {output_path}", flush=True)
        
        conn = duckdb.connect(database=":memory:")
        
        # Load Parquet file into DuckDB
        conn.execute(f"CREATE TABLE data AS SELECT * FROM parquet_scan('{file_path}')")
        print(f"Loaded {file_path}", flush=True)
        
        # Use IN clause for efficient batch searching
        placeholders = ','.join(['?'] * len(search_values))
        query = f"SELECT word_pair, frequency FROM data WHERE word_pair IN ({placeholders})"
        result = conn.execute(query, search_values).fetch_df()
        print(f"Finished query for {file_path}", flush=True)
        
        conn.close()
        
        output_path = Path(output_path) / f"{Path(file_path).name.split('.')[0]}_llava_format_freqs.csv"
        # Save results to output file
        if not result.empty:
            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(output_path, index=False)
            print(f"Saved {len(result)} results to {output_path}", flush=True)
        else:
            print(f"No results found!!", flush=True)
            
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}", flush=True)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freqs_path", type=str, required=True)
    parser.add_argument("--pairs_path", type=str, required=True)

    args = parser.parse_args()
    pairs_df = pd.read_csv(args.pairs_path)
    pairs_df['pairs'] = pairs_df['pairs'].apply(lambda x: eval(x.replace(" ", ",")))
    all_pairs = [x for y in pairs_df['pairs'].values for x in y if isinstance(x, str)]
    all_pairs = [x.replace(",", " ") for x in all_pairs]
    all_pairs = pd.unique(all_pairs)
    print(all_pairs[:5])
    
    print(f"searching for {len(all_pairs)} unique pairs from {args.freqs_path} in {Path(args.freqs_path).name}")
    if "freq" not in Path(args.pairs_path).name:
        get_freqs_for_file(all_pairs, args.freqs_path, Path(args.pairs_path).parent / "llava_format_freqs_pred")
    else:
        freq_bin = "_".join(Path(args.pairs_path).name.split("_")[:2])
        get_freqs_for_file(all_pairs, args.freqs_path, Path(args.pairs_path).parent / f"{freq_bin}_full_caption_freqs")
