import argparse
import os
import time
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)

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

def score_row(sia, text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {"sentiment_neg": 0.0, "sentiment_neu": 1.0, "sentiment_pos": 0.0, "sentiment_compound": 0.0}
    scores = sia.polarity_scores(text)
    return {"sentiment_neg": scores["neg"], "sentiment_neu": scores["neu"], "sentiment_pos": scores["pos"], "sentiment_compound": scores["compound"]}

def main():
    start_time = time.time()
    print(f"[sentiment] Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    args = parse_args()
    df = load_parquet_folder(args.data)
    print(f"[sentiment] Loaded {len(df):,} rows")
    sia = SentimentIntensityAnalyzer()
    text_col = "reviewText_normalized" if "reviewText_normalized" in df.columns else "reviewText"
    print(f"[sentiment] Scoring column: {text_col}")
    scores = df[text_col].apply(lambda t: score_row(sia, t))
    scores_df = pd.DataFrame(scores.tolist())
    df = pd.concat([df, scores_df], axis=1)
    print(f"[sentiment] compound mean={df['sentiment_compound'].mean():.4f}")
    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)
    print(f"[sentiment] Written -> {args.out}")
    elapsed = time.time() - start_time
    print(f"[sentiment] End: {time.strftime('%Y-%m-%d %H:%M:%S')}  Elapsed: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
