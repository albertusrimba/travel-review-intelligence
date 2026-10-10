
from pathlib import Path

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


# 1. Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "balanced_dataset.csv"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load and deduplicate the dataset
df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates(subset=["review"]).copy()

X = df["review"]
y = df["label"]

print("Dataset size after deduplication:", len(df))
print("\nLabel distribution:")
print(y.value_counts())


# 3. Split the data before fitting the model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# 4. Build the baseline pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            max_features=20000,
            ngram_range=(1, 2),
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


# 5. Train the model using training data only
print("\nTraining baseline model...")
model.fit(X_train, y_train)


# 6. Predict on the held-out test set
y_pred = model.predict(X_test)


# 7. Evaluate model performance
print("\n" + "=" * 50)
print("BASELINE MODEL RESULTS")
print("=" * 50)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Macro F1-score: {f1_score(y_test, y_pred, average='macro'):.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))


# 8. Create and save the confusion matrix
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

plt.title("Baseline Model - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

confusion_path = FIGURE_DIR / "baseline_confusion_matrix.png"
plt.savefig(confusion_path, dpi=150)
plt.close()

print("\nConfusion matrix saved to:")
print(confusion_path.relative_to(PROJECT_ROOT))

print("\nBaseline training and evaluation completed.")
