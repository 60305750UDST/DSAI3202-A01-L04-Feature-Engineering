# DSAI3202 — Lab 4: Text Feature Engineering with Azure ML

**Course:** DSAI3202 - Cloud Computing for Data Science and AI  
**Semester:** Winter 2026  
**Student ID:** 60305750  
**Branch:** `lab4_feature_engineering`

---

## Overview

This lab implements a full 7-component text feature engineering pipeline for Amazon Electronics product reviews using Azure Machine Learning. The pipeline processes a sampled Gold dataset (~210k reviews) and generates structured numerical features — including length features, VADER sentiment scores, TF-IDF sparse vectors, and SBERT dense embeddings — for downstream machine learning tasks.

---

## Repository Structure
```
DSAI3202-A01-L04-Feature-Engineering/
├── components/
│   ├── split_dataset/              # Train/val/test split (70/15/15)
│   ├── normalize_text/             # Text cleaning and normalization
│   ├── length_features/            # Word and character count features
│   ├── sentiment_features/         # VADER sentiment scores
│   ├── tfidf_features/             # TF-IDF sparse matrix (500 features)
│   ├── sbert_embeddings/           # SBERT dense embeddings (all-MiniLM-L6-v2)
│   └── merge_features/             # Merges all feature outputs into one parquet
├── environments/
│   ├── sentiment/                  # conda.yml + env.yml (nltk + pandas)
│   ├── tfidf/                      # conda.yml + env.yml (scikit-learn + scipy)
│   └── sbert/                      # conda.yml + env.yml (PyTorch + sentence-transformers)
├── data/
│   └── features_v1_sampled.yml     # Azure ML data asset definition
├── datastores/
│   └── curated_adls.yml            # Azure Blob datastore pointing to curated container
├── feature_store/
│   ├── entity_amazon_review.yml    # Feature store entity (asin + reviewerID)
│   ├── FeatureSetSpec.yaml         # Feature set specification with source path
│   └── feature_set_amazon_review_text_features.yml
├── pipelines/
│   └── feature_pipeline.yml        # Full 9-step pipeline definition
└── README.md
```

---

## Infrastructure

| Resource        | Name                                       |
|-----------------|--------------------------------------------|
| Resource Group  | `rg-60305750`                              |
| ML Workspace    | `Amazon-Electronics-Lab-60305750`          |
| Compute Cluster | `cpu-cluster` (Standard_DS3_v2, 0–2 nodes) |
| Storage Account | `amazondatalake60305750`                   |
| Container       | `curated`                                  |
| Data Asset      | `amazon_electronics_features_v1_sampled:1` |
| Datastore       | `blobkey`                                  |
| Feature Store   | `amazon-electronics-fs-60305750`           |

---

## Pipeline Components

### 1. `split_dataset`

Splits the input parquet data into train (70%), validation (15%), and test (15%) sets using `sklearn.model_selection.train_test_split` with `seed=42`.

### 2. `normalize_text`

Cleans raw review text: lowercases, removes URLs, numbers, and punctuation, then filters out reviews shorter than 10 characters. Applied independently to train, val, and test splits.

### 3. `length_features`

Extracts two numerical features per review:

- `review_length_words` — word count
- `review_length_chars` — character count

### 4. `sentiment_features`

Generates four sentiment scores per review using VADER (`nltk.sentiment.vader.SentimentIntensityAnalyzer`) on the `reviewText_normalized` column:

- `sentiment_neg`
- `sentiment_neu`
- `sentiment_pos`
- `sentiment_compound`

### 5. `tfidf_features`

Fits a `sklearn.TfidfVectorizer` on the training split only (`max_features=500`, `stop_words='english'`), then transforms train, val, and test splits. Outputs three separate parquet files.

### 6. `sbert_embeddings`

Generates 384-dimensional dense semantic embeddings per review using `sentence-transformers` with the `all-MiniLM-L6-v2` model (`batch_size=64`).

### 7. `merge_features`

Merges all feature outputs (length, sentiment, TF-IDF train, SBERT train) into a single unified feature matrix. Final output: **209,980 rows × 896 columns** (683.6 MB parquet).

---

## Custom Environments

Three custom environments were registered and built before running the pipeline:

| Environment     | Key Dependencies                              | Build Status |
|-----------------|-----------------------------------------------|--------------|
| `sentiment-env:1` | nltk==3.8.1, pandas==1.5.3, pyarrow==12.0.1 | ✅ Succeeded |
| `tfidf-env:1`   | scikit-learn==1.3.2, pandas, scipy==1.10.1   | ✅ Succeeded |
| `sbert-env:1`   | torch==2.0.1, sentence-transformers==2.7.0, transformers==4.40.0 | ✅ Succeeded |

All environments use `mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest` as the base Docker image.

> **Lesson learned:** Azure ML CLI v2 requires a base Docker image when creating custom environments. A conda file alone is not sufficient — the environment must be wrapped in an Azure ML environment YAML spec that specifies `image:` alongside `conda_file:`, then registered with `az ml environment create --file env.yml`.

---

## Setup & Deployment

### Prerequisites

- Azure ML CLI v2 (`az ml`)
- Access to workspace `Amazon-Electronics-Lab-60305750`

### One-time Setup
```bash
# Set defaults
az configure --defaults group="rg-60305750" workspace="Amazon-Electronics-Lab-60305750"

# Register datastore
az ml datastore create --file datastores/curated_adls.yml

# Register environments
az ml environment create --file environments/sentiment/env.yml
az ml environment create --file environments/tfidf/env.yml
az ml environment create --file environments/sbert/env.yml

# Register all components
az ml component create --file components/split_dataset/component.yml
az ml component create --file components/normalize_text/component.yml
az ml component create --file components/length_features/component.yml
az ml component create --file components/sentiment_features/component.yml
az ml component create --file components/tfidf_features/component.yml
az ml component create --file components/sbert_embeddings/component.yml
az ml component create --file components/merge_features/component.yml
```

### Run the Pipeline
```bash
az ml job create --file pipelines/feature_pipeline.yml
```

### Monitor
```bash
az ml job show --name <job_name> --query status
az ml job stream --name <job_name>
```

---

## Successful Pipeline Run

![Completed Pipeline](screenshots/successful_pipeline.png)

| Job Name                    | Status          |
|-----------------------------|-----------------|
| `brave_turtle_pjb5rrqgr3`   | **Completed** ✅ |

All 9 steps completed successfully:
```
split → normalize_train / normalize_val / normalize_test
      → length_train + sentiment_train + sbert_train + tfidf
      → merge_all
```

**Final output:** 209,980 rows × 896 features, written as `merged.parquet` (683.6 MB).

---

## Feature Store

- **Feature Store:** `amazon-electronics-fs-60305750` (Qatar Central)
- **Entity:** `AmazonReview:1` (index columns: `asin`, `reviewerID`)
- **Feature Set:** `amazon_review_text_features:1`
- Defined in `feature_store/entity_amazon_review.yml` and `feature_store/FeatureSetSpec.yaml`
```bash
# Register entity
az ml feature-store-entity create \
  --file feature_store/entity_amazon_review.yml \
  --resource-group rg-60305750 \
  --feature-store-name amazon-electronics-fs-60305750

# Register feature set
az ml feature-set create \
  --file feature_store/feature_set_amazon_review_text_features.yml \
  --resource-group rg-60305750 \
  --feature-store-name amazon-electronics-fs-60305750
```
