
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "balanced_dataset.csv"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Load the original dataset
df = pd.read_csv(DATA_PATH)

# Exact duplicate reviews have already been checked
# for conflicting labels in the previous audit.
df_unique = df.drop_duplicates(subset=["review"]).copy()

# Calculate review length
df_unique["review_length"] = df_unique["review"].str.len()

print("=" * 50)
print("DATASET BEFORE AND AFTER DEDUPLICATION")
print("=" * 50)

print("Original rows:", len(df))
print("Unique review rows:", len(df_unique))
print("Rows removed:", len(df) - len(df_unique))

print("\nOriginal label distribution:")
print(df["label"].value_counts())

print("\nUnique review label distribution:")
print(df_unique["label"].value_counts())

print("\nUnique review label percentages:")
print(
    (df_unique["label"].value_counts(normalize=True) * 100).round(2)
)

print("\n" + "=" * 50)
print("REVIEW LENGTH STATISTICS")
print("=" * 50)

print(df_unique["review_length"].describe().round(2))

# Use a consistent visual style
sns.set_theme(style="whitegrid")

# 1. Label distribution after deduplication
plt.figure(figsize=(7, 5))
sns.countplot(data=df_unique, x="label")
plt.title("Sentiment Distribution After Deduplication")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()

label_plot = FIGURE_DIR / "sentiment_distribution.png"
plt.savefig(label_plot, dpi=150)
plt.close()

# 2. Review length distribution
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df_unique,
    x="review_length",
    bins=40
)
plt.title("Review Length Distribution")
plt.xlabel("Review Length (Characters)")
plt.ylabel("Number of Reviews")
plt.tight_layout()

length_plot = FIGURE_DIR / "review_length_distribution.png"
plt.savefig(length_plot, dpi=150)
plt.close()

print("\nSaved visualizations:")
print(label_plot.relative_to(PROJECT_ROOT))
print(length_plot.relative_to(PROJECT_ROOT))

print("\nEDA completed.")
