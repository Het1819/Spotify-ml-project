import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score

def train():
    # 1. Load Data
    df = pd.read_csv('spotify_2015_2025_85k.csv')
    
    # 2. Feature Engineering (From your notebook)
    df['is_popular'] = (df['popularity'] >= 60).astype(int)
    # Drop identifier columns
    drop_cols = ['track_id', 'track_name', 'artist_name', 'album_name', 'release_date', 'popularity']
    X = df.drop(columns=drop_cols + ['is_popular'])
    y = df['is_popular']

    # 3. Define Preprocessing Pipeline
    # Identify categorical and numerical columns
    cat_cols = ['genre', 'label', 'country'] # Based on your get_dummies logic
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            # handle_unknown='ignore' prevents crashes on new genres in production
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )

    # 4. Create Model Pipeline
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(n_estimators=300, learning_rate=0.05))
    ])

    # 5. Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)

    # 6. Evaluate
    print(f"ROC-AUC: {roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])}")

    # 7. Save Artifacts
    joblib.dump(clf, 'models/spotify_pipeline.pkl')
    print("Model saved to models/spotify_pipeline.pkl")

if __name__ == "__main__":
    train()