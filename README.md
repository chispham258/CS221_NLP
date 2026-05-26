# Tweet Emotion Classification — `nlp-experimental-1.ipynb`

Fine-tuning transformer models on the [TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) emotion task with a CE vs CE+SCL loss ablation and cross-model disagreement analysis.

**Task:** 4-class emotion classification on tweets — `anger`, `joy`, `optimism`, `sadness`.

---

## Notebook Structure

`nlp-experimental-1.ipynb` is organized into 8 sections. Each section is a logical component described below.

---

### 1. Setup & Imports (Cells 0–2)

Installs dependencies (commented out, uncomment if needed), imports libraries, and sets global constants:

```python
TASK = "emotion"
SEED = 42
```

All random seeds are fixed (`torch`, `numpy`, `random`, `transformers.set_seed`) for reproducibility. Device is auto-detected (`cuda` or `cpu`).

---

### 2. Dataset (Cells 3–7)

Loads and preprocesses TweetEval emotion data from HuggingFace:

```python
dataset = load_dataset("cardiffnlp/tweet_eval", "emotion")
```

| Split      | Size  |
|------------|-------|
| train      | 3,257 |
| validation | 374   |
| test       | 1,421 |

**Label IDs:** `0=anger`, `1=joy`, `2=optimism`, `3=sadness`

**Tweet preprocessing** is applied to all splits before tokenization:

```python
def preprocess_tweet(text):
    # @mention → @user  (anonymize usernames)
    # http://... → http  (collapse all URLs)
```

This follows the CardiffNLP convention used when pretraining Twitter-specific models. Reduces out-of-vocabulary fragmentation for non-Twitter checkpoints.

A label distribution plot for the training set is also displayed here. `optimism` is the minority class (~14% of training data).

---

### 3. Loss Functions — `SupConLoss` and `CESCLTrainer` (Cells 8–10)

Two classes are defined here:

**`SupConLoss`** — Supervised Contrastive Loss (Khosla et al., 2020):
- Input: L2-normalized CLS token embeddings, shape `[batch, 1, hidden_dim]`
- Pulls same-class embeddings together, pushes different-class apart
- Parameters: `temperature=0.3`, `base_temperature=0.3`

**`CESCLTrainer`** — custom HuggingFace `Trainer` subclass:
- Computes CE loss from `AutoModelForSequenceClassification` output
- Extracts CLS token from last hidden layer, normalizes it, runs `SupConLoss`
- Combines: `L = L_CE + scl_weight × L_SCL` (`scl_weight=0.1`)

---

### 4. Model Configurations (Cells 11–12)

Two config lists define what gets trained:

**`MODEL_CONFIGS`** — 7 Twitter-pretrained models, all trained with CE only:

| Short name | HuggingFace checkpoint |
|------------|------------------------|
| Rob-rt     | `roberta-base` |
| Rob-tw     | `cardiffnlp/twitter-roberta-base` |
| XLM-r      | `xlm-roberta-base` |
| XLM-tw     | `cardiffnlp/twitter-xlm-roberta-base` |
| BERTweet   | `vinai/bertweet-base` |
| TimeLM-19  | `cardiffnlp/twitter-roberta-base-2019-90m` |
| TimeLM-21  | `cardiffnlp/twitter-roberta-base-2021-124m` |

**`ROB_BS_CONFIGS`** — roberta-base loss ablation:

| Short name    | Training mode | scl_weight | scl_temperature |
|---------------|--------------|------------|-----------------|
| Rob-bs-CE     | CE only      | —          | —               |
| Rob-bs-CE-SCL | CE + SCL     | 0.1        | 0.3             |

> Note: `Rob-bs-CE-SCL` is commented out by default. Uncomment in `ROB_BS_CONFIGS` to train it.

**Shared hyperparameters for all fine-tuned models:**

```
lr           = 1e-5
batch_size   = 16
epochs       = 5
max_length   = 128
weight_decay = 0.01
```

---

### 5. Tokenization (Cell 13)

`get_tokenizer(model_name)` handles BERTweet specially — it requires a slow tokenizer with built-in normalization:

```python
# BERTweet: use_fast=False, normalization=True
# All others: use_fast=True
```

`tokenize_dataset()` maps tokenization over all splits with `truncation=True, max_length=128`. Dynamic padding is applied per batch via `DataCollatorWithPadding`.

---

### 6. Metrics (Cells 14–15)

`compute_metrics(eval_pred)` is passed to `Trainer` and computes on the validation/test set after each epoch:

| Metric            | Description |
|-------------------|-------------|
| `accuracy`        | Overall accuracy |
| `macro_f1`        | Primary metric — unweighted F1 across all 4 classes |
| `weighted_f1`     | F1 weighted by class support |
| `macro_precision` | Unweighted precision |
| `macro_recall`    | Unweighted recall |

`extract_logits()` handles the case where `CESCLTrainer` returns `(logits, hidden_states, ...)` tuples instead of bare logits.

---

### 7. Training Pipeline (Cells 16–17)

`train_one_model(config, dataset, label_names)` is the main training function. It:

1. Tokenizes the dataset for the given model
2. Loads `AutoModelForSequenceClassification` from HuggingFace
3. Runs `Trainer` (or `CESCLTrainer` for CE+SCL) for `epochs=5`
4. Evaluates on validation and test splits
5. Generates a confusion matrix plot
6. Saves model + tokenizer to `saved_models/`
7. Writes all metrics and reports to `results/`
8. Frees GPU memory with `torch.cuda.empty_cache()` + `gc.collect()`

---

### 8. Training (Cells 18–22)

The actual training invocations. Several are commented out — this is intentional so you can run only the cells you need:

| Cell | What it does | State |
|------|-------------|-------|
| 19   | Train `ROB_BS_CONFIGS` (Rob-bs-CE, Rob-bs-CE-SCL) | commented out |
| 20   | Eval Rob-bs untrained baseline (no fine-tuning) | commented out |
| 21   | Cross-model disagreement analysis | **active** |
| 22   | Train full `MODEL_CONFIGS` sweep | commented out |

**Cell 21 — Confusing Examples Analysis** is the only active training cell. It loads 4 pre-trained models from `saved_models/` and runs inference on the test set:

```python
CONFUSION_MODELS = [
    {"short_name": "rob_rt",     "path": "./saved_models/emotion_Rob-rt"},
    {"short_name": "rob_tw",     "path": "./saved_models/emotion_Rob-tw"},
    {"short_name": "rob_ce",     "path": "./saved_models/emotion_Rob-bs-CE"},
    {"short_name": "rob_ce_cls", "path": "./saved_models/emotion_Rob-bs-CE-SCL"},
]
```

For each test example it counts how many unique predictions the 4 models give (`n_unique`). Examples with `n_unique >= 3` are "confusing" — models strongly disagree.

---

### 9. Results (Cells 23–25)

Commented-out cells for:
- Aggregating all results into a sorted DataFrame by macro F1
- Saving to `results/emotion_all_model_results.csv`
- Comparing our scores vs paper baselines (CardiffNLP TweetEval paper)

Uncomment these after running training to see the full comparison table.

---

## How to Run

### Prerequisites

```bash
pip install datasets transformers accelerate scikit-learn \
            pandas matplotlib seaborn torch
```

Python 3.10+. GPU strongly recommended (training uses `fp16` automatically on CUDA).

### Run the notebook

```bash
jupyter notebook nlp-experimental-1.ipynb
```

**To run the full sweep from scratch:**

1. Uncomment the loop in Cell 22 (`MODEL_CONFIGS` sweep)
2. Uncomment the loop in Cell 19 (`ROB_BS_CONFIGS` — CE and CE+SCL)
3. Optionally uncomment Cell 20 for the untrained `Rob-bs` baseline
4. Run all cells top to bottom

**To only run the confusing-example analysis** (requires saved models):

- Skip Cells 19, 20, 22
- Run Cell 21 directly — it loads from `saved_models/`

---

## Outputs

After training, files are written to two directories:

### `saved_models/emotion_<short_name>/`

Full model checkpoint — tokenizer + weights. Can be loaded with:

```python
AutoTokenizer.from_pretrained("saved_models/emotion_Rob-bs-CE")
AutoModelForSequenceClassification.from_pretrained("saved_models/emotion_Rob-bs-CE")
```

### `results/`

| File | Contents |
|------|----------|
| `emotion_<model>_metrics.csv` | Single-row CSV: accuracy, macro_f1, weighted_f1, precision, recall, eval_loss, train_time_sec, hyperparams |
| `emotion_<model>_classification_report.csv` | Per-class precision/recall/F1/support + macro/weighted averages |
| `emotion_<model>_confusion_matrix.csv` | 4×4 confusion matrix (true vs predicted label) |
| `emotion_<model>_confusion_matrix.png` | Heatmap visualization of the confusion matrix |
| `emotion_all_model_results.csv` | All models aggregated (written by Cell 24 after full sweep) |
| `emotion_confusing_examples.csv` | Test examples where ≥3 of 4 models disagree, sorted by disagreement count |

### `outputs/emotion/<short_name>/`

Intermediate HuggingFace `Trainer` checkpoints written during training (not the final model).

---

## Demo App

`app.py` is a Gradio web UI that loads 4 roberta-base variants from `saved_models/` and runs inference interactively. Requires the saved models to exist.

```bash
python app.py
# → http://localhost:7860  (+ public share link)
```

| Dropdown | Model dir |
|----------|-----------|
| Rob-bs         | `saved_models/emotion_Rob-bs` |
| Rob-tw         | `saved_models/emotion_Rob-tw` |
| Rob-bs-CE      | `saved_models/emotion_Rob-bs-CE` |
| Rob-bs-CE-SCL  | `saved_models/emotion_Rob-bs-CE-SCL` |
