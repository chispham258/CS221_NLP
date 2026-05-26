import os
import tempfile
os.environ.setdefault("GRADIO_TEMP_DIR", os.path.join(tempfile.gettempdir(), f"gradio_{os.getuid()}"))
os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)

import torch
import gradio as gr
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

MODEL_MAP = {
    "Rob-bs": "emotion_Rob-bs",
    "Rob-tw": "emotion_Rob-tw",
    "Rob-bs-CE": "emotion_Rob-bs-CE",
    "Rob-bs-CE-SCL": "emotion_Rob-bs-CE-SCL",
}

LABELS = ["anger", "joy", "optimism", "sadness"]

LABEL_EMOJI = {
    "anger": "😡",
    "joy": "😄",
    "optimism": "🌟",
    "sadness": "😢",
}

_cache: dict = {}


def preprocess_tweet(text: str) -> str:
    tokens = []
    for t in text.split():
        if len(t) > 1:
            t = "@user" if t.startswith("@") and t.count("@") == 1 else t
            t = "http" if t.startswith("http") else t
        tokens.append(t)
    return " ".join(tokens)


def load_model(short_name: str):
    if short_name in _cache:
        return _cache[short_name]
    model_dir = os.path.join(SAVED_MODELS_DIR, MODEL_MAP[short_name])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    _cache[short_name] = (tokenizer, model)
    return tokenizer, model


def classify(text: str, model_name: str):
    if not text.strip():
        return "Please enter some text.", {}

    tokenizer, model = load_model(model_name)
    processed = preprocess_tweet(text)

    inputs = tokenizer(
        processed,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).squeeze().numpy()
    pred_idx = int(np.argmax(probs))
    pred_label = LABELS[pred_idx]
    emoji = LABEL_EMOJI[pred_label]

    label_probs = {label: float(probs[i]) for i, label in enumerate(LABELS)}

    return f"{emoji} {pred_label.upper()}", label_probs


with gr.Blocks(title="Tweet Emotion Classifier") as demo:
    gr.Markdown("# Tweet Emotion Classifier")
    gr.Markdown(
        "Classify the emotion of a tweet into one of four categories: "
        "**anger**, **joy**, **optimism**, **sadness**.\n\n"
        "Models fine-tuned on [TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) emotion task."
    )

    with gr.Row():
        with gr.Column(scale=2):
            model_dropdown = gr.Dropdown(
                choices=list(MODEL_MAP.keys()),
                value="Rob-bs-CE-SCL",
                label="Model",
            )
            text_input = gr.Textbox(
                lines=3,
                placeholder="Type or paste a tweet...",
                label="Tweet",
            )
            submit_btn = gr.Button("Classify", variant="primary")

        with gr.Column(scale=3):
            label_output = gr.Label(label="Predicted Emotion")
            bar_output = gr.BarPlot(
                x="emotion",
                y="confidence",
                label="Confidence Scores",
                y_lim=[0, 1],
                color="emotion",
            )

    gr.Examples(
        examples=[
            ["I can't believe they did that, absolutely furious right now!!!", "Rob-bs-CE-SCL"],
            ["Just got the job offer, this is the best day ever!", "Rob-bs-CE-SCL"],
            ["Things will get better, I just know it :)", "Rob-bs-CE-SCL"],
            ["Missing everyone so much, wish things were different...", "Rob-bs-CE-SCL"],
        ],
        inputs=[text_input, model_dropdown],
    )

    def run(text, model_name):
        pred, probs = classify(text, model_name)
        import pandas as pd
        df = pd.DataFrame({"emotion": list(probs.keys()), "confidence": list(probs.values())})
        return pred, df

    submit_btn.click(
        fn=run,
        inputs=[text_input, model_dropdown],
        outputs=[label_output, bar_output],
    )
    text_input.submit(
        fn=run,
        inputs=[text_input, model_dropdown],
        outputs=[label_output, bar_output],
    )

if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Soft())
