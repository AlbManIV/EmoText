import argparse, os
import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments)
import numpy as np
from sklearn.model_selection import train_test_split
import evaluate

def main(args):
    df = pd.read_csv(args.csv)
    df[args.text_col] = df[args.text_col].astype(str)
    df[args.label_col] = df[args.label_col].astype(str)

    # Label encoding
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df[args.label_col])
    id2label = {i: lab for i, lab in enumerate(le.classes_)}
    label2id = {lab: i for i, lab in id2label.items()}

    # Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label_id"])

    # HF datasets
    ds_train = Dataset.from_pandas(train_df[[args.text_col, "label_id"]].rename(columns={args.text_col:"text","label_id":"label"}))
    ds_test  = Dataset.from_pandas(test_df[[args.text_col, "label_id"]].rename(columns={args.text_col:"text","label_id":"label"}))

    # Tokenizer & model (BETO)
    model_name = "dccuchile/bert-base-spanish-wwm-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(le.classes_), id2label=id2label, label2id=label2id)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    ds_train_tok = ds_train.map(tokenize, batched=True)
    ds_test_tok = ds_test.map(tokenize, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Metrics
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"]
        }

    os.makedirs(args.out_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        push_to_hub=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train_tok,
        eval_dataset=ds_test_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Save artifacts
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    # Save label mapping
    with open(os.path.join(args.out_dir, "labels.txt"), "w", encoding="utf-8") as f:
        for lab in le.classes_:
            f.write(lab + "\n")

    print(f"Modelo guardado en {args.out_dir}. Etiquetas:", list(le.classes_))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out_dir", type=str, default="outputs/beto-emotext")
    args = parser.parse_args()
    main(args)
