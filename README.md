# DSAI 3202 — Lab 4  
## Text Feature Engineering with Azure ML (Part 1)

This project implements a complete **text feature engineering pipeline using Azure Machine Learning** on the Amazon Electronics Reviews dataset.  
The goal of this lab is to build reusable feature engineering components, construct a pipeline, and register features in the Azure ML Feature Store.

---

# Project Overview

The pipeline processes raw review text and generates multiple feature types:

- Text normalization
- Length-based features
- Sentiment analysis features
- TF-IDF vector representations
- Sentence embeddings using SBERT
- Final merged feature dataset

These features can later be used for downstream machine learning models such as **review rating prediction or sentiment classification**.

---

# Dataset

Dataset used:

**Amazon Electronics Reviews**

The dataset contains customer reviews including:

- `reviewText`
- `overall` (rating)
- `reviewerID`
- `asin`
- timestamps and metadata

The pipeline processes the `reviewText` field to generate numerical features.

---

# Feature Engineering Pipeline

The Azure ML pipeline contains the following steps:

1. **Split Dataset**
   - Train / Validation / Test split

2. **Text Normalization**
   - Lowercasing
   - Removing punctuation
   - Removing extra whitespace

3. **Length Features**
   - Character length
   - Word count

4. **Sentiment Features**
   - VADER sentiment analysis
   - Outputs:
     - `sentiment_neg`
     - `sentiment_neu`
     - `sentiment_pos`
     - `sentiment_compound`

5. **TF-IDF Features**
   - Converts text into sparse word importance vectors

6. **SBERT Embeddings**
   - Uses Sentence-BERT to produce dense semantic embeddings

7. **Feature Merge**
   - Combines all engineered features into a single dataset

Final dataset shape:

```
(209980 rows, 896 features)
```

---

# Project Structure

```
lab4_feature_engineering/
│
├── components/
│   ├── normalize_text/
│   ├── length_features/
│   ├── sentiment_features/
│   ├── tfidf_features/
│   ├── sbert_embeddings/
│   └── merge_features/
│
├── environments/
│   ├── sentiment-env
│   ├── tfidf-env
│   └── sbert-env
│
├── pipelines/
│   └── feature_engineering_pipeline.py
│
├── feature_store/
│   └── amazon_review_text_features
│
├── data/
│   └── amazon_electronics_reviews
│
└── README.md
```

---

# Feature Store

A feature store was created in Azure ML to manage reusable features.

### Feature Store
```
amazon-electronics-fs
```

### Entity
```
AmazonReview
```

### Feature Set
```
amazon_review_text_features
```

Features registered include:

- `review_length_words`
- `review_length_chars`
- `sentiment_neg`
- `sentiment_neu`
- `sentiment_pos`
- `sentiment_compound`

---

# Azure ML Environments

Separate environments were created for modular pipeline execution:

- `sentiment-env`
- `tfidf-env`
- `sbert-env`

Each environment includes the required dependencies for its component.

---

# Final Output

The pipeline generates a merged feature dataset:

```
merged.parquet
```

Size:

```
~683 MB
```

This dataset contains all engineered features ready for model training.

---

# Technologies Used

- Python
- Azure Machine Learning
- Scikit-learn
- Sentence Transformers (SBERT)
- VADER Sentiment Analysis
- Pandas
- TF-IDF Vectorization

---

# Learning Objectives

This lab demonstrates:

- Modular ML pipeline construction
- Feature engineering for NLP
- Azure ML component development
- Azure ML Feature Store usage
- Scalable text processing pipelines
