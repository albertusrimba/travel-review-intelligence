import unittest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


class TestSentimentAPI(unittest.TestCase):
    def test_health_endpoint(self):
        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

    def test_positive_review_prediction(self):
        response = client.post(
            "/predict",
            json={
                "review": "The hotel was clean and the staff were very helpful."
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sentiment"], "POSITIVE")

    def test_negative_review_prediction(self):
        response = client.post(
            "/predict",
            json={
                "review": "The room was dirty and the service was terrible."
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sentiment"], "NEGATIVE")

    def test_empty_review_is_rejected(self):
        response = client.post(
            "/predict",
            json={"review": ""},
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
