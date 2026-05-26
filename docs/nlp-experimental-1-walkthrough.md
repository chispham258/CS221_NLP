# NLP Experiment 1 — TweetEval Emotion Classification Walkthrough

This document walks through `nlp-experimental-1.ipynb`, a fine-tuning experiment
on the TweetEval emotion task. It covers the dataset, preprocessing, model
configurations, loss functions, training pipeline, evaluation metrics, and results.

---

## 1. Task & Dataset

**Task:** Multi-class emotion classification on tweets.
**Dataset:** [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval), `emotion` subset.

**Labels (4 classes):**

| ID | Label    |
|----|----------|
| 0  | anger    |
| 1  | joy      |
| 2  | optimism |
| 3  | sadness  |

Loaded via HuggingFace Datasets:

```python
from datasets import load_dataset
dataset = load_dataset("cardiffnlp/tweet_eval", "emotion")
```

Split sizes (approximate):

| Split | Size |
|-------|------|
| train | 3,257 |
| validation | 374 |
| test | 1,421 |
