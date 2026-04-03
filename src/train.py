import argparse
import os
import glob
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import joblib
import mlflow

def load_data(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {folder_path}")
    return pd.read_parquet(files[0])

def create_labels(df):
    df['label'] = (df['overall'] >= 4).astype(int)
    return df

def build_features(df):
    sbert_cols = sorted([col for col in df.columns if col.startswith('sbert_')])
    tfidf_cols = sorted([col for col in df.columns if col.startswith('tfidf_')])
    sentiment_cols = [col for col in ['sentiment_pos', 'sentiment_neg', 'sentiment_neu', 'sentiment_compound'] if col in df.columns]
    length_cols = [col for col in ['review_length_words', 'review_length_chars'] if col in df.columns]
    feature_cols = sbert_cols + tfidf_cols + sentiment_cols + length_cols
    print(f"Using {len(feature_cols)} features")
    return df[feature_cols].fillna(0).values

def evaluate(model, X, y, split):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    mlflow.log_metric(f"{split}_accuracy", accuracy_score(y, y_pred))
    mlflow.log_metric(f"{split}_auc", roc_auc_score(y, y_proba))
    mlflow.log_metric(f"{split}_precision", precision_score(y, y_pred, zero_division=0))
    mlflow.log_metric(f"{split}_recall", recall_score(y, y_pred, zero_division=0))
    mlflow.log_metric(f"{split}_f1", f1_score(y, y_pred, zero_division=0))
    print(f"{split} accuracy: {accuracy_score(y, y_pred):.4f}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.000814)
    parser.add_argument("--max_iter", type=int, default=2000)
    return parser.parse_args()

def main():
    start_time = time.time()
    args = parse_args()
    mlflow.log_param("alpha", args.alpha)
    mlflow.log_param("max_iter", args.max_iter)

    print("Loading data...")
    train_df = create_labels(load_data(args.train_data))
    val_df = create_labels(load_data(args.val_data))
    test_df = create_labels(load_data(args.test_data))

    print("Building features...")
    X_train = build_features(train_df)
    X_val = build_features(val_df)
    X_test = build_features(test_df)
    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    print("Training model...")
    model = LogisticRegression(C=1/args.alpha, max_iter=args.max_iter, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("Evaluating...")
    evaluate(model, X_train, y_train, "train")
    evaluate(model, X_val, y_val, "val")
    evaluate(model, X_test, y_test, "test")

    print("Saving model...")
    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, "model.pkl")
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)
    mlflow.log_metric("training_runtime_seconds", time.time() - start_time)
    print("Done.")

if __name__ == "__main__":
    main()
