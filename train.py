import gzip
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

DATA_DIR = "reference_repo/dataset"
CLICKBAIT_PATH = os.path.join(DATA_DIR, "clickbait_data.gz")
NON_CLICKBAIT_PATH = os.path.join(DATA_DIR, "non_clickbait_data.gz")


def load_lines(file_path):
    """Read headlines line-by-line from a gzipped text file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f]


def load_dataset():
    """Load clickbait and news headlines into a cleaned DataFrame."""
    cb_lines = load_lines(CLICKBAIT_PATH)
    ncb_lines = load_lines(NON_CLICKBAIT_PATH)

    df_cb = pd.DataFrame({"headline": cb_lines, "label": 1})
    df_ncb = pd.DataFrame({"headline": ncb_lines, "label": 0})

    df = pd.concat([df_cb, df_ncb], ignore_index=True)

    # Drop empty rows and duplicates
    df = df[df["headline"].str.strip() != ""]
    df = df.drop_duplicates(subset=["headline"]).reset_index(drop=True)
    return df


def main():
    print("Loading data...")
    df = load_dataset()
    print(f"Loaded {len(df)} total headlines after cleaning.")

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        df["headline"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    # Scikit-learn Pipeline: TF-IDF (word n-grams 1-2) + Logistic Regression
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    print("Training pipeline...")
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["News (0)", "Clickbait (1)"]
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Clickbait Headline Detector")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved confusion matrix to confusion_matrix.png")

    # Save model artifact
    joblib.dump(pipeline, "model.joblib")
    print("Saved trained pipeline to model.joblib")


if __name__ == "__main__":
    main()
