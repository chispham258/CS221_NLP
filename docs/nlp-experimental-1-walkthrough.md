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

---

## 2. Tweet Preprocessing

Tweets are normalized before tokenization to reduce noise from usernames and URLs,
following the CardiffNLP convention used when pretraining Twitter-specific models.

```python
def preprocess_tweet(text):
    new_text = []
    for t in text.split():
        if len(t) > 1:
            t = "@user" if t.startswith("@") and t.count("@") == 1 else t
            t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)
```

**Rules:**
- `@mention` (single `@`) → `@user` — anonymizes usernames, reduces vocabulary size
- `http://...` or `https://...` → `http` — collapses all URLs to a single token

Applied to all splits via `dataset.map(add_preprocessed_text, batched=True)` before tokenization.

**Example:**

| Original | Preprocessed |
|----------|-------------|
| `@john I love this! https://t.co/abc` | `@user I love this! http` |
| `RT @news breaking story http://bit.ly/x` | `RT @user breaking story http` |
