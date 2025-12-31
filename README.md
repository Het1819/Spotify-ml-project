# Spotify Song Popularity Predictor 🎵

A machine learning project that predicts whether a Spotify song will be "popular" based on its audio features and metadata. This project includes a comprehensive data analysis notebook, a training pipeline using LightGBM, and an interactive web application built with Streamlit and FastAPI.

**Live Demo:** [Spotify ML Project App](https://spotify-ml-project-k3d6pzoqbqlopz8zplmvrj.streamlit.app/)

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Usage](#installation--usage)
- [Model Performance](#model-performance)

## 🚀 Project Overview
The goal of this project is to classify songs as **Popular** or **Not Popular** (Binary Classification). 
A song is defined as "Popular" if its popularity score is **≥ 60**.

The solution involves:
1.  **Data Analysis**: Exploratory Data Analysis (EDA) on 85k songs from 2015-2025.
2.  **Preprocessing**: Handling categorical variables (Genre, Label, Country) and scaling numerical features.
3.  **Modeling**: Training a `LGBMClassifier` (LightGBM) model wrapped in a Scikit-Learn pipeline.
4.  **Deployment**: A Streamlit frontend that communicates with an internal FastAPI backend to serve predictions.

## ✨ Features
* **Interactive Web Interface**: Adjust song parameters (Danceability, Energy, Tempo, etc.) and see real-time predictions.
* **Hybrid App Architecture**: Combines Streamlit for UI and FastAPI for inference logic within a single application.
* **Robust Preprocessing**: Pipelines handle unknown categories gracefully using OneHotEncoder with `handle_unknown='ignore'`.
* **Docker Support**: Includes `Dockerfile` and DevContainer configuration for consistent development environments.

## 📊 Dataset
The model is trained on `spotify_2015_2025_85k.csv`.
* **Input Features**:
    * **Numerical**: `duration_ms`, `danceability`, `energy`, `key`, `loudness`, `mode`, `instrumentalness`, `tempo`, `stream_count`, `explicit`.
    * **Categorical**: `genre`, `label`, `country`.
* **Target**: `is_popular` (derived from `popularity` column).

## 🛠️ Tech Stack
* **Language**: Python 3.11
* **Machine Learning**: LightGBM, Scikit-Learn
* **Data Processing**: Pandas, NumPy
* **Web Frameworks**: Streamlit, FastAPI
* **Visualization**: Matplotlib, Seaborn
* **Containerization**: Docker

## 📂 Project Structure
```text
spotify-ml-project/
├── .devcontainer/       # Dev container configuration
├── models/
│   └── spotify_pipeline.pkl  # Trained model pipeline
├── src/
│   ├── main_app.py      # Streamlit + FastAPI application entry point
│   └── train.py         # Script to retrain the model
├── Dockerfile           # Docker image definition
├── requirements.txt     # Python dependencies
├── Spotify.ipynb        # EDA and experimentation notebook
├── spotify_2015_2025_85k.csv # Dataset
└── README.md            # Project documentation

## 📊 Strategic Insights & Business Analysis

Beyond model performance, this project aimed to uncover behavioral drivers behind user listening habits. By analyzing audio features against popularity metrics, several key consumer insights emerged:

### 1. The "Energy-Engagement" Correlation
Exploratory Data Analysis (EDA) revealed a strong positive correlation between **Energy** and **Loudness** with **Track Popularity** within the Pop and Hip-Hop genres. 
* **Insight:** Users in these segments prioritize high-tempo, dynamic content.
* **Business Application:** For a streaming platform, "Cold Start" recommendations for new users in these demographics should prioritize tracks with Energy scores > 0.7 to maximize initial retention.

### 2. Impact of "Acousticness" on Niche Markets
While "Acousticness" generally showed a negative correlation with global popularity, it was the strongest predictor of engagement within the "Indie" and "Folk" sub-genres.
* **Conclusion:** One-size-fits-all algorithms fail here. Feature weighting must be dynamic based on the user's preferred genre cluster to avoid irrelevant recommendations.
