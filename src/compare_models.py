
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "balanced_dataset.csv"

df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates(subset=["review"]).copy()

X = df["review"]
y = df["label"]

# 60% train, 40% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y,
)

# Split the temporary data equally into validation and test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Final test samples:", len(X_test))

experiments = {
    "unigram": (1, 1),
    "unigram_bigram": (1, 2),
}

results = []

for experiment_name, ngram_range in experiments.items():
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                max_features=20000,
                ngram_range=ngram_range,
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

    print(f"\nTraining experiment: {experiment_name}")
    model.fit(X_train, y_train)

    # Use validation data for model comparison
    val_predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, val_predictions)
    macro_f1 = f1_score(
        y_val,
        val_predictions,
        average="macro",
    )

    results.append({
        "experiment": experiment_name,
        "ngram_range": str(ngram_range),
        "validation_accuracy": accuracy,
        "validation_macro_f1": macro_f1,
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "validation_macro_f1",
    ascending=False,
)

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)
print(results_df.to_string(index=False))

print("\nComparison completed.")
print("The final test set has not been used for evaluation.")
