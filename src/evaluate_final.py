
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "balanced_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# 1. Load and deduplicate the dataset
df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates(subset=["review"]).copy()

X = df["review"]
y = df["label"]


# 2. Recreate the exact same split used for comparison
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)


# 3. Combine training and validation data
X_train_final = pd.concat(
    [X_train, X_val],
    ignore_index=True,
)
y_train_final = pd.concat(
    [y_train, y_val],
    ignore_index=True,
)

print("Final training samples:", len(X_train_final))
print("Final test samples:", len(X_test))


# 4. Build the selected unigram model
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            max_features=20000,
            ngram_range=(1, 1),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
    ),
])


# 5. Train on training + validation only
print("\nTraining selected model...")
model.fit(X_train_final, y_train_final)


# 6. Evaluate once on the held-out test set
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
)

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0,
)

print("\n" + "=" * 50)
print("FINAL TEST RESULTS")
print("=" * 50)

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    digits=4,
    zero_division=0,
))


# 7. Save the confusion matrix
labels = ["NEGATIVE", "POSITIVE"]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels,
)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
)

plt.title("Final Model - Test Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

confusion_path = FIGURE_DIR / "final_confusion_matrix.png"
plt.savefig(confusion_path, dpi=150)
plt.close()


# 8. Save the model and evaluation results
model_path = MODEL_DIR / "sentiment_model.joblib"
joblib.dump(model, model_path)

metrics = {
    "model": "TF-IDF + Logistic Regression",
    "ngram_range": [1, 1],
    "deduplicated_rows": len(df),
    "training_rows": len(X_train_final),
    "test_rows": len(X_test),
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "classification_report": report,
}

metrics_path = REPORT_DIR / "final_metrics.json"
with metrics_path.open("w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

print("\nSaved model:", model_path.relative_to(PROJECT_ROOT))
print("Saved metrics:", metrics_path.relative_to(PROJECT_ROOT))
print("Saved confusion matrix:", confusion_path.relative_to(PROJECT_ROOT))
print("\nFinal evaluation completed.")
