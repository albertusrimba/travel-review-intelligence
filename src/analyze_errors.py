
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "balanced_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load and deduplicate the data
df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates(subset=["review"]).copy()

X = df["review"]
y = df["label"]

# Recreate exactly the same split as the baseline
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Recreate the baseline model
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

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Collect incorrectly classified reviews
errors = pd.DataFrame({
    "review": X_test.reset_index(drop=True),
    "actual_label": y_test.reset_index(drop=True),
    "predicted_label": predictions,
})

errors = errors[
    errors["actual_label"] != errors["predicted_label"]
].copy()

errors["review_length"] = errors["review"].str.len()

# Keep the original test-set row order
output_path = OUTPUT_DIR / "misclassified_reviews.csv"
errors.to_csv(output_path, index=False)

print("Misclassified reviews:", len(errors))
print("\nError counts by actual and predicted label:")
print(
    errors.groupby(["actual_label", "predicted_label"])
    .size()
)

print("\nSample misclassified reviews:")
for _, row in errors.head(10).iterrows():
    print("-" * 60)
    print("Actual:", row["actual_label"])
    print("Predicted:", row["predicted_label"])
    print("Length:", row["review_length"])
    print("Review:", row["review"][:500])

print("\nSaved error analysis to:", output_path)
