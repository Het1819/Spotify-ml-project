from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained pipeline
model = joblib.load('models/spotify_pipeline.pkl')

app = FastAPI(title="Spotify Hit Predictor")

# Define input data schema based on your CSV columns
class SongInput(BaseModel):
    danceability: float
    energy: float
    key: int
    loudness: float
    mode: int
    instrumentalness: float
    tempo: float
    stream_count: int
    explicit: int
    genre: str
    country: str
    label: str
    duration_ms: int
    # Note: We exclude release_date/names as they were dropped in training

@app.post("/predict")
def predict(song: SongInput):
    # Convert input to DataFrame
    data = pd.DataFrame([song.dict()])
    
    # Predict probability
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    
    return {
        "is_popular": int(prediction),
        "hit_probability": float(probability),
        "status": "Potential Hit" if prediction == 1 else "Niche Track"
    }