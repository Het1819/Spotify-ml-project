# 🎵 Spotify Hit Predictor

A machine learning-powered application that predicts whether a song will be a "Hit" or a "Niche Track" based on its audio features and metadata. This project demonstrates an end-to-end ML deployment using **Streamlit** for the frontend and **FastAPI** for the backend logic.

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://spotify-ml-project-k3d6pzoqbqlopz8zplmvrj.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688)](https://fastapi.tiangolo.com/)

## 🚀 Live Demo
Check out the live application here:  
**[Spotify Hit Predictor App](https://spotify-ml-project-k3d6pzoqbqlopz8zplmvrj.streamlit.app/)**

---

## 🧐 About The Project
This tool helps artists and producers analyze the potential popularity of a track before its release. By inputting audio features like danceability, energy, and tempo, the model predicts the likelihood of the song becoming a hit.

### Key Features
* **Interactive UI**: Built with Streamlit for easy data input and visualization.
* **Dual-Architecture**: Combines a Streamlit frontend with a FastAPI backend in a single application using `TestClient`.
* **Real-time Inference**: Instant predictions using a pre-trained LightGBM/Scikit-learn pipeline.
* **Probability Metrics**: Displays the exact probability of a song being a hit.

---

## 🛠️ Tech Stack
* **Frontend**: Streamlit
* **Backend**: FastAPI
* **Machine Learning**: Scikit-learn, LightGBM, Joblib
* **Data Manipulation**: Pandas
* **Container/Env**: Python venv (or Docker if applicable)

---

## 📂 Directory Structure
```text
spotify-ml-project/
├── models/
│   └── spotify_pipeline.pkl    # Trained ML pipeline
├── src/
│   ├── main_app.py             # Main application entry point (Streamlit + FastAPI)
│   └── train.py                # Script for training the model
├── requirements.txt            # Project dependencies
├── app.py                      # Standalone API entry point (optional)
└── README.md                   # Project documentation
