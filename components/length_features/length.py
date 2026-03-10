import argparse
import os
import time
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()

def load_parquet_folder(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {folder_path}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

def main():
    start_time = time.time()
    print(f"[length] Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    args = parse_args()
    df = load_parquet_folder(args.data)
    print(f"[length] Loaded {len(df):,} rows")
    df = df.copy()
    text_series = df["reviewText"].fillna("")
    df["review_length_words"] = text_series.str.split().str.len().fillna(0).astype(int)
    df["review_length_chars"] = text_series.str.len().fillna(0).astype(int)
    print(f"[length] Words mean={df['review_length_words'].mean():.1f}  Chars mean={df['review_length_chars'].mean():.1f}")
    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)
    print(f"[length] Written -> {args.out}")
    elapsed = time.time() - start_time
    print(f"[length] End: {time.strftime('%Y-%m-%d %H:%M:%S')}  Elapsed: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
