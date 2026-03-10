import argparse
import os
import re
import string
import time
import pandas as pd

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b\d+\b")
_MULTI_SPACE = re.compile(r"\s+")

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _NUMBER_RE.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text

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
    print(f"[normalize] Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    args = parse_args()
    df = load_parquet_folder(args.data)
    print(f"[normalize] Loaded {len(df):,} rows")
    df = df.copy()
    df["reviewText_normalized"] = df["reviewText"].apply(normalize_text)
    before = len(df)
    df = df[df["reviewText_normalized"].str.len() >= 10].reset_index(drop=True)
    print(f"[normalize] Removed {before - len(df):,} short reviews. Remaining: {len(df):,}")
    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)
    print(f"[normalize] Written -> {args.out}")
    elapsed = time.time() - start_time
    print(f"[normalize] End: {time.strftime('%Y-%m-%d %H:%M:%S')}  Elapsed: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
