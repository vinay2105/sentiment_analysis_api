from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

app = FastAPI()

class ReviewRequest(BaseModel):
    review: str

@app.get("/")
async def root():
    return {
        "messege" : "Welcome to sentiment analysis APIuse the endpoint /Review"
    }


@app.post("Review")
async def Review(request: ReviewRequest):
    review = request.review
    analysis = sentiment_pipeline(review)[0]
    result = {
        "review": review,
        "sentiment": analysis['label'],
        "confidence": round(analysis['score'], 2)
    }
    return result

