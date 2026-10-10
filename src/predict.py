
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_model.joblib"


def load_model():
    """Load the trained sentiment classification pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. "
            "Run src/evaluate_final.py first."
        )

    return joblib.load(MODEL_PATH)


def predict_sentiment(review: str, model=None) -> dict:
    """Predict sentiment for a single review."""
    if not isinstance(review, str):
        raise TypeError("Review must be a string.")

    if not review.strip():
        raise ValueError("Review cannot be empty.")

    if model is None:
        model = load_model()

    prediction = model.predict([review])[0]

    result = {
        "review": review,
        "sentiment": str(prediction),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([review])[0]
        classes = model.classes_

        result["probabilities"] = {
            str(label): round(float(probability), 4)
            for label, probability in zip(classes, probabilities)
        }

    return result


if __name__ == "__main__":
    examples = [
        "The hotel was clean and the staff were very helpful.",
        "The room was dirty and the service was terrible.",
        "The hotel was okay, but the location was not convenient.",
    ]

    model = load_model()

    for review in examples:
        result = predict_sentiment(review, model)
        print("-" * 60)
        print("Review:", result["review"])
        print("Predicted sentiment:", result["sentiment"])

        if "probabilities" in result:
            print("Probabilities:", result["probabilities"])
