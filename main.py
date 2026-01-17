# ======================================
# PHISHING URL DETECTION - API (FINAL FIX)
# ======================================

import os
import re
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import urlparse
from scipy.sparse import hstack, csr_matrix

# -----------------------------
# APP CONFIGURATION
# -----------------------------
app = FastAPI(title="Phishing URL Guard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# LOAD MODEL ARTIFACTS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(BASE_DIR, "phishing_cnb_model.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(BASE_DIR, "char_tfidf_vectorizer.pkl"), "rb") as f:
        char_vectorizer = pickle.load(f)
except FileNotFoundError:
    model = None
    char_vectorizer = None

# -----------------------------
# CONSTANTS
# -----------------------------
PHISHING_THRESHOLD = 0.75  # reduces false positives

# -----------------------------
# PREPROCESSING
# -----------------------------
def clean_url(url: str) -> str:
    """
    Normalize URL for TF-IDF:
    - keep domain + path
    - normalize numbers, hashes, UUID-like tokens
    """
    url = url.lower().strip()
    parsed = urlparse(url if url.startswith("http") else "http://" + url)

    domain = parsed.netloc.replace("www.", "")
    path = parsed.path

    # Normalize dynamic tokens
    path = re.sub(r"\d+", "<num>", path)
    path = re.sub(r"[a-f0-9]{8,}", "<hash>", path)

    return domain + path


def url_features(urls: pd.Series) -> csr_matrix:
    """
    Numeric features extracted ONLY from domain
    (prevents sub-path false positives)
    """
    features = []

    for url in urls:
        parsed = urlparse(url if url.startswith("http") else "http://" + url)
        domain = parsed.netloc.replace("www.", "")

        features.append([
            len(domain),                  # domain length
            domain.count("."),            # number of subdomains
            sum(c.isdigit() for c in domain),
            domain.count("-")             # hyphen count
        ])

    return csr_matrix(np.array(features, dtype=np.float32))

# -----------------------------
# API MODELS
# -----------------------------
class URLRequest(BaseModel):
    url: str

# -----------------------------
# ROUTES
# -----------------------------
@app.get("/")
def root():
    return {"message": "Phishing URL Guard API is running"}


@app.post("/predict")
def predict_url(request: URLRequest):
    if model is None or char_vectorizer is None:
        raise HTTPException(
            status_code=500,
            detail="Model files not loaded properly"
        )

    url = request.url.strip()

    # Prepare dataframe
    df = pd.DataFrame({"url": [url]})
    df["url_clean"] = df["url"].apply(clean_url)

    try:
        # Feature extraction
        X_char = char_vectorizer.transform(df["url_clean"])
        X_num = url_features(df["url"])
        X_final = hstack([X_char, X_num])

        # Prediction
        probs = model.predict_proba(X_final)
        phishing_index = list(model.classes_).index(1)
        phishing_prob = float(probs[0, phishing_index])

        return {
            "url": url,
            "prediction": "phishing"
            if phishing_prob >= PHISHING_THRESHOLD
            else "benign",
            "phishing_probability": round(phishing_prob, 4)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

# -----------------------------
# LOCAL DEVELOPMENT
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
