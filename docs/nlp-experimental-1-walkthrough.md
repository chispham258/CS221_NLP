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

---

## 3. Label Distribution

The notebook plots the training set label distribution to understand class balance.

```python
train_labels = dataset["train"]["label"]
df_dist = pd.DataFrame({
    "label_id": train_labels,
    "label": [label_names[i] for i in train_labels]
})
sns.countplot(data=df_dist, x="label", order=label_names)
```

**Training set distribution (approximate):**

| Label    | Count |
|----------|-------|
| anger    | ~1,600 |
| joy      | ~1,100 |
| optimism | ~700  |
| sadness  | ~900  |

`optimism` is the minority class (~14% of training data). This class-imbalance
explains why all models show lower per-class F1 on `optimism` relative to the other labels.

**Test set support (from classification reports):**

| Label    | Test samples |
|----------|-------------|
| anger    | 558 |
| joy      | 358 |
| optimism | 123 |
| sadness  | 382 |

---

## 4. Model Configurations

Four models are evaluated, covering an untrained baseline, a Twitter-pretrained model,
and two loss-function ablations on `roberta-base`.

| Short name | HuggingFace checkpoint | Training mode | Notes |
|------------|------------------------|--------------|-------|
| Rob-bs | `roberta-base` | none (untrained) | Zero-shot baseline — no fine-tuning |
| Rob-tw | `cardiffnlp/twitter-roberta-base` | CE | RoBERTa pretrained on tweets, standard fine-tuning |
| Rob-bs-CE | `roberta-base` | CE | roberta-base fine-tuned with cross-entropy only |
| Rob-bs-CE-SCL | `roberta-base` | CE + SCL | roberta-base fine-tuned with CE + contrastive loss |

Shared fine-tuning hyperparameters (all except Rob-bs):

```python
lr = 1e-5
batch_size = 16
epochs = 5
max_length = 128
```

`Rob-bs-CE-SCL` additional params: `scl_weight=0.1`, `scl_temperature=0.3`.

**Design intent:** Rob-bs establishes a random-chance ceiling. Rob-tw shows the effect
of Twitter-domain pretraining. Rob-bs-CE vs Rob-bs-CE-SCL isolates the effect of
adding supervised contrastive loss on top of CE.

---

## 5. Loss Functions

### 5.1 Standard Cross-Entropy (CE)

Default for all standard fine-tuning experiments. The HuggingFace `Trainer` computes CE
loss automatically from `AutoModelForSequenceClassification`.

### 5.2 Supervised Contrastive Loss (SCL)

Implemented as `SupConLoss`, a re-implementation of the SupCon paper
(Khosla et al., 2020), operating on CLS-token representations.

```python
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.3, base_temperature=0.3):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        # features: [batch_size, n_views, hidden_dim]
        # Pulls same-class representations together,
        # pushes different-class representations apart.
        ...
```

**Key properties:**
- `temperature=0.3` — controls sharpness of the similarity distribution
- `n_views=1` — no augmentation; single CLS embedding per sample
- Uses cosine similarity (L2-normalized features) scaled by temperature

### 5.3 Combined CE + SCL (`CESCLTrainer`)

A custom `Trainer` subclass that computes both losses and combines them:

```python
class CESCLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        ce_loss = outputs.loss

        # CLS token from last hidden layer
        features = outputs.hidden_states[-1][:, 0, :]
        features = F.normalize(features, p=2, dim=1).unsqueeze(1)

        scl_loss = self.scl_loss_fn(features, inputs["labels"])
        loss = ce_loss + self.scl_weight * scl_loss
        return (loss, outputs) if return_outputs else loss
```

**Combined loss:** `L = L_CE + α · L_SCL` where `α = scl_weight = 0.1`

The low `α` keeps CE as the dominant signal while SCL regularizes the representation
space to be more class-discriminative.

---

## 6. Training Pipeline

### 6.1 Tokenization

```python
def get_tokenizer(model_name):
    if model_name == "vinai/bertweet-base":
        # BERTweet requires its own slow tokenizer with built-in normalization
        return AutoTokenizer.from_pretrained(model_name, use_fast=False, normalization=True)
    else:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
```

All models in this experiment use `roberta-base` or `cardiffnlp/twitter-roberta-base`,
so `use_fast=True` applies throughout. All inputs truncated to `max_length=128` tokens.
Dynamic padding via `DataCollatorWithPadding`.

### 6.2 HuggingFace Trainer

Standard `TrainingArguments` for CE-only models:

```python
TrainingArguments(
    output_dir=f"./outputs/emotion/{short_name}",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=1e-5,
    seed=42,
)
```

`Rob-bs-CE-SCL` uses `CESCLTrainer` (same args) instead of the default `Trainer`.
`Rob-bs` uses no Trainer — it is evaluated zero-shot without fine-tuning.

### 6.3 Reproducibility

Seed fixed to `42` via `set_seed(42)` (transformers), `random.seed`, `np.random.seed`,
and `torch.manual_seed` before each model training run.
