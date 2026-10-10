
import unittest

from src.predict import load_model, predict_sentiment


class TestSentimentPrediction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the model once for all tests
        cls.model = load_model()

    def test_positive_review(self):
        review = "The hotel was clean and the staff were very helpful."

        result = predict_sentiment(review, self.model)

        self.assertEqual(result["sentiment"], "POSITIVE")

    def test_negative_review(self):
        review = "The room was dirty and the service was terrible."

        result = predict_sentiment(review, self.model)

        self.assertEqual(result["sentiment"], "NEGATIVE")

    def test_prediction_contains_probabilities(self):
        result = predict_sentiment(
            "The hotel was clean and comfortable.",
            self.model,
        )

        self.assertIn("probabilities", result)
        self.assertAlmostEqual(
            sum(result["probabilities"].values()),
            1.0,
            places=4,
        )

    def test_empty_review_is_rejected(self):
        with self.assertRaises(ValueError):
            predict_sentiment("   ", self.model)

    def test_non_string_review_is_rejected(self):
        with self.assertRaises(TypeError):
            predict_sentiment(123, self.model)

    def test_prediction_has_valid_label(self):
        result = predict_sentiment(
            "The hotel was okay.",
            self.model,
        )

        self.assertIn(
            result["sentiment"],
            ["POSITIVE", "NEGATIVE"],
        )


if __name__ == "__main__":
    unittest.main()
