from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from predict import predict

# ------------------ APP INIT ------------------
app = FastAPI(
    title="AI Review Intelligence API",
    description="API for sentiment analysis using a trained machine learning model",
    version="1.0"
)

# ------------------ REQUEST MODEL ------------------
class Review(BaseModel):
    text: str

# ------------------ ROOT REDIRECT ------------------
@app.get("/")
def root():
    return RedirectResponse(url="/docs")

# ------------------ PREDICTION ENDPOINT ------------------
@app.post("/predict")
def get_sentiment(review: Review):
    
    # Input validation
    if not review.text or review.text.strip() == "":
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    result = predict(review.text)

    # Handle internal errors
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result