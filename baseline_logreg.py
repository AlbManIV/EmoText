import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib

def main(args):
    df = pd.read_csv(args.csv)
    texts = df[args.text_col].astype(str).tolist()
    labels = df[args.label_col].astype(str).tolist()

    le = LabelEncoder()
    y = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=args.test_size, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=30000)),
        ("clf", LogisticRegression(max_iter=200))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    print("Accuracy:", round(acc, 4))
    print("F1-macro:", round(f1m, 4))
    print("\nReporte de clasificación:\n", classification_report(y_test, y_pred, target_names=le.classes_))

    out_dir = "outputs/baseline_logreg"
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pipe, os.path.join(out_dir, "model.joblib"))
    joblib.dump(le, os.path.join(out_dir, "label_encoder.joblib"))
    print(f"\nModelo guardado en {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()
    main(args)
