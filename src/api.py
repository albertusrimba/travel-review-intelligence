from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import load_model, predict_sentiment

app = FastAPI(
    title="Travel Review Intelligence API",
    description="API for hotel review sentiment prediction",
    version="1.0.0",
)

model = load_model()


class ReviewRequest(BaseModel):
    review: str = Field(min_length=1, description="Hotel review text")


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: ReviewRequest):
    try:
        result = predict_sentiment(request.review, model)
        return result
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error))