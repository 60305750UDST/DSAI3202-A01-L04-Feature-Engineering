import argparse
import os
import time
import pandas as pd

ENTITY_KEYS = ["asin", "reviewerID"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=str, required=True)
    parser.add_argument("--sentiment", type=str, required=True)
    parser.add_argument("--tfidf", type=str, required=True)
    parser.add_argument("--sbert", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()

def load_parquet_folder(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {folder_path}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

def keep_feature_cols(df, prefix):
    feat_cols = [c for c in df.columns if c.startswith(prefix)]
    keys = [k for k in ENTITY_KEYS if k in df.columns]
    return df[keys + feat_cols]

def main():
    start_time = time.time()
    print(f"[merge] Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    args = parse_args()

    length_df = load_parquet_folder(args.length)
    sentiment_df = load_parquet_folder(args.sentiment)
    tfidf_df = load_parquet_folder(args.tfidf)
    sbert_df = load_parquet_folder(args.sbert)

    print(f"[merge] Shapes: length={length_df.shape} sentiment={sentiment_df.shape} tfidf={tfidf_df.shape} sbert={sbert_df.shape}")

    meta_cols = ["overall", "reviewText", "reviewText_normalized", "summary", "unixReviewTime"]
    base_cols = ENTITY_KEYS + [c for c in meta_cols if c in length_df.columns] + ["review_length_words", "review_length_chars"]
    base_df = length_df[[c for c in base_cols if c in length_df.columns]].copy()

    merged = base_df.merge(keep_feature_cols(sentiment_df, "sentiment_"), on=ENTITY_KEYS, how="left")
    merged = merged.merge(keep_feature_cols(tfidf_df, "tfidf_"), on=ENTITY_KEYS, how="left")
    merged = merged.merge(keep_feature_cols(sbert_df, "sbert_"), on=ENTITY_KEYS, how="left")

    num_cols = [c for c in merged.columns if any(c.startswith(p) for p in ["review_length_", "sentiment_", "tfidf_", "sbert_"])]
    merged[num_cols] = merged[num_cols].fillna(0.0)

    print(f"[merge] Final shape: {merged.shape}")
    os.makedirs(args.out, exist_ok=True)
    merged.to_parquet(os.path.join(args.out, "merged.parquet"), index=False)
    print(f"[merge] Written -> {args.out}")

    elapsed = time.time() - start_time
    print(f"[merge] End: {time.strftime('%Y-%m-%d %H:%M:%S')}  Elapsed: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
