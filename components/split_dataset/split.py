import argparse
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    return parser.parse_args()

def load_parquet_folder(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {folder_path}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

def main():
    start_time = time.time()
    print(f"[split] Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    args = parse_args()

    df = load_parquet_folder(args.data)
    print(f"[split] Loaded {len(df):,} rows")

    train_df, temp_df = train_test_split(df, test_size=(1 - args.train_ratio), random_state=args.seed, shuffle=True)
    val_relative = args.val_ratio / (1 - args.train_ratio)
    val_df, test_df = train_test_split(temp_df, test_size=(1 - val_relative), random_state=args.seed, shuffle=True)

    print(f"[split] Train={len(train_df):,}  Val={len(val_df):,}  Test={len(test_df):,}")

    for df_out, path, name in [(train_df, args.train_out, "train"), (val_df, args.val_out, "val"), (test_df, args.test_out, "test")]:
        os.makedirs(path, exist_ok=True)
        df_out.to_parquet(os.path.join(path, "data.parquet"), index=False)
        print(f"[split] Written {name} -> {path}")

    elapsed = time.time() - start_time
    print(f"[split] End: {time.strftime('%Y-%m-%d %H:%M:%S')}  Elapsed: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
