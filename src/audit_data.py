
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "balanced_dataset.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("1. LABEL DISTRIBUTION")
print("=" * 50)

label_counts = df["label"].value_counts(dropna=False)
print(label_counts)

print("\nLabel percentages:")
print((df["label"].value_counts(normalize=True) * 100).round(2))

print("\n" + "=" * 50)
print("2. DUPLICATE ANALYSIS")
print("=" * 50)

duplicate_mask = df.duplicated(subset=["review"], keep=False)
duplicate_reviews = df.loc[duplicate_mask].sort_values("review")

print("Rows involved in repeated reviews:", duplicate_mask.sum())
print("Unique review texts:", df["review"].nunique())
print("Repeated review groups:")
print(df["review"].value_counts().loc[lambda x: x > 1].head(10))

print("\n" + "=" * 50)
print("3. LABEL CONSISTENCY")
print("=" * 50)

labels_per_review = df.groupby("review")["label"].nunique()
conflicting_reviews = labels_per_review[labels_per_review > 1]

print("Review texts with conflicting labels:", len(conflicting_reviews))

if not conflicting_reviews.empty:
    print("\nExamples of conflicting labels:")
    print(
        df[df["review"].isin(conflicting_reviews.index)]
        .sort_values("review")
        .head(10)
        .to_string(index=False)
    )

print("\n" + "=" * 50)
print("4. TEXT QUALITY")
print("=" * 50)

print("Empty or whitespace-only reviews:", 
      df["review"].fillna("").str.strip().eq("").sum())

print("\nReview length statistics:")
print(df["review"].str.len().describe())

print("\nAudit completed.")
