# DSAI3202 Assignment 2 - Model Training & Automation

## Model Choice
Logistic Regression (scikit-learn)

## Features Used
- SBERT embeddings (768 dims)
- TF-IDF vectors
- Sentiment features (pos, neg, neu, compound)
- Review length features

## Hyperparameter Tuning
Sweep job with 6 trials (random sampling):
- Best: alpha=0.000814, max_iter=2000
- Best val_accuracy: 0.87666

## Final Performance
- Train accuracy: ~0.90
- Val accuracy: 0.877
- Test accuracy: 0.876
