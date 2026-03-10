import argparse
import os
import pickle
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    parser.add_argument("--max_features", type=int, default=500)
    return parser.parse_args()

def load_parquet_folder(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {folder_path}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

def get_text(df):
    col = "reviewText_normalized" if "reviewText_normalized" in df.columns else "reviewText"
    return df[col].fillna("").astype(str)

def main():
    start_time = time.time()
    print(f"[tfidf] Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    args = parse_args()

    train_df = load_parquet_folder(args.train)
    val_df = load_parquet_folder(args.val)
    test_df = load_parquet_folder(args.test)
    print(f"[tfidf] Rows: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

    vectorizer = TfidfVectorizer(max_features=args.max_features, stop_words="english", ngram_range=(1,2), sublinear_tf=True, dtype="float32")
    print(f"[tfidf] Fitting on train...")
    train_matrix = vectorizer.fit_transform(get_text(train_df))
    val_matrix = vectorizer.transform(get_text(val_df))
    test_matrix = vectorizer.transform(get_text(test_df))

    feature_names = [f"tfidf_{i}" for i in range(train_matrix.shape[1])]

    for df_src, matrix, out_path, name in [
        (train_df, train_matrix, args.train_out, "train"),
        (val_df, val_matrix, args.val_out, "val"),
        (test_df, test_matrix, args.test_out, "test")
    ]:
        tfidf_df = pd.DataFrame(matrix.toarray(), columns=feature_names)
        out_df = pd.concat([df_src.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
        os.makedirs(out_path, exist_ok=True)
        out_df.to_parquet(os.path.join(out_path, "data.parquet"), index=False)
        print(f"[tfidf] Written {name} -> {out_path}")

    with open(os.path.join(args.train_out, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    elapsed = time.time() - start_time
    print(f"[tfidf] End: {time.strftime('%Y-%m-%d %H:%M:%S')}  Elapsed: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
