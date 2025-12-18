import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.testclient import TestClient

# ==========================================
# 1. MODEL INITIALIZATION (The Engine)
# ==========================================
# Professional path handling
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'spotify_pipeline.pkl')

# Load model once at startup (Singleton pattern)
@st.cache_resource
def load_spotify_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_spotify_model()
except Exception as e:
    st.error(f"Failed to load model from {MODEL_PATH}. Ensure the path is correct!")
    st.stop()

# ==========================================
# 2. BACKEND LOGIC (The API Layer)
# ==========================================
app = FastAPI()

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

@app.post("/predict")
def predict_popularity(song: SongInput):
    data = pd.DataFrame([song.dict()])
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    
    return {
        "is_popular": int(prediction),
        "hit_probability": float(probability),
        "status": "Potential Hit" if prediction == 1 else "Niche Track"
    }

# We use TestClient to communicate between the UI and the API internally
client = TestClient(app)

# ==========================================
# 3. FRONTEND UI (The Streamlit Layer)
# ==========================================
st.set_page_config(page_title="Spotify Hit Predictor", page_icon="🎵", layout="wide")

st.title("🎵 Spotify Hit Predictor")
st.markdown("### Professional ML Deployment: End-to-End System")
st.info("This app combines a FastAPI backend with a Streamlit frontend using a internal TestClient.")

# UI Layout
col1, col2 = st.columns(2)

with col1:
    st.header("Track Metadata")
    genre = st.selectbox("Genre", ["Pop", "Rock", "Hip-Hop", "Jazz", "Metal", "Indie", "Reggaeton"])
    country = st.selectbox("Country", ["US", "UK", "Canada", "Germany", "Brazil", "France", "Mexico"])
    label = st.selectbox("Label", ["Universal Music", "Sony Music", "Warner Music", "Indie", "Island Records"])
    duration_ms = st.number_input("Duration (ms)", value=200000)
    stream_count = st.number_input("Current Streams", value=50000, step=1000)

with col2:
    st.header("Audio Features")
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
    energy = st.slider("Energy", 0.0, 1.0, 0.5)
    loudness = st.slider("Loudness (dB)", -60.0, 0.0, -10.0)
    tempo = st.number_input("Tempo (BPM)", value=120.0)
    explicit = st.radio("Explicit Content", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)

# Prediction Logic
if st.button("🚀 Analyze Track Popularity"):
    payload = {
        "danceability": danceability, "energy": energy, "key": 5, "loudness": loudness,
        "mode": 1, "instrumentalness": 0.01, "tempo": tempo, "stream_count": stream_count,
        "explicit": explicit, "genre": genre, "country": country, "label": label, "duration_ms": duration_ms
    }

    with st.spinner('🤖 Model is performing inference...'):
        # Internal call from UI to API
        response = client.post("/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            st.divider()
            
            # Display Results
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                if result["is_popular"] == 1:
                    st.success(f"### Result: {result['status']}")
                else:
                    st.warning(f"### Result: {result['status']}")
            
            with m_col2:
                st.metric("Hit Probability", f"{result['hit_probability']:.2%}")
        else:
            st.error("API Error: Unable to get prediction.")