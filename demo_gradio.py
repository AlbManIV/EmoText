import argparse, os
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

def load_labels(model_dir):
    labels_path = os.path.join(model_dir, "labels.txt")
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = [l.strip() for l in f if l.strip()]
    else:
        labels = None
    return labels

def build_pipeline(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    labels = load_labels(model_dir)
    if labels is None:
        labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    def predict(text):
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            result = {labels[i]: float(probs[i]) for i in range(len(labels))}
            return labels[idx], result
    return predict, labels

def main(args):
    predict, labels = build_pipeline(args.model_dir)
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Textbox(label="Escribe un texto en español"),
        outputs=[gr.Label(label="Emoción predicha"), gr.Label(label="Probabilidades")],
        title="EmoText — Detección de emociones",
        description="Modelo BETO fine-tuned. Clases: " + ", ".join(labels)
    )
    demo.launch()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="outputs/beto-emotext")
    args = parser.parse_args()
    main(args)
