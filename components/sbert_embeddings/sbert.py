import argparse
import os
import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()

def load_parquet_folder(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {folder_path}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

def main():
    start_time = time.time()
    print(f"[sbert] Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    args = parse_args()
    df = load_parquet_folder(args.data)
    print(f"[sbert] Loaded {len(df):,} rows")
    text_col = "reviewText_normalized" if "reviewText_normalized" in df.columns else "reviewText"
    texts = df[text_col].fillna("").astype(str).tolist()
    print(f"[sbert] Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name)
    print(f"[sbert] Encoding {len(texts):,} reviews...")
    embeddings = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True)
    print(f"[sbert] Embedding shape: {embeddings.shape}")
    emb_cols = [f"sbert_{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings.astype("float32"), columns=emb_cols)
    out_df = pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
    os.makedirs(args.out, exist_ok=True)
    out_df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)
    print(f"[sbert] Written -> {args.out}")
    elapsed = time.time() - start_time
    print(f"[sbert] End: {time.strftime('%Y-%m-%d %H:%M:%S')}  Elapsed: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
