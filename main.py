# ======================================
# PHISHING URL DETECTION - API (STABLE)
# ======================================

import os
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
PHISHING_THRESHOLD = 0.75

# -----------------------------
# FAMOUS / TRUSTED DOMAINS
# -----------------------------
FAMOUS_DOMAINS = {
    "google.com",
    "youtube.com",
    "gmail.com",
    "github.com",
    "stackoverflow.com",
    "chatgpt.com",
    "openai.com",
    "huggingface.co",
    "amazon.com",
    "apple.com",
    "microsoft.com",
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "twitter.com",
    "x.com",
    "whatsapp.com"
}

# -----------------------------
# HELPERS
# -----------------------------
def extract_domain(url: str) -> str:
    url = url.lower().strip()
    parsed = urlparse(url if url.startswith("http") else "http://" + url)
    domain = parsed.netloc
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def is_famous_domain(domain: str) -> bool:
    """
    Matches:
    - google.com
    - www.google.com
    - mail.google.com
    """
    for famous in FAMOUS_DOMAINS:
        if domain == famous or domain.endswith("." + famous):
            return True
    return False


def url_features(domains: pd.Series) -> csr_matrix:
    features = []

    for domain in domains:
        features.append([
            len(domain),
            domain.count("."),
            sum(c.isdigit() for c in domain),
            domain.count("-")
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
        raise HTTPException(500, "Model files not loaded")

    url = request.url.strip()
    domain = extract_domain(url)

    # ✅ FAMOUS DOMAIN OVERRIDE
    if is_famous_domain(domain):
        return {
            "url": url,
            "domain": domain,
            "prediction": "benign",
            "reason": "famous_domain",
            "phishing_probability": 0.0
        }

    # -----------------------------
    # ML PREDICTION (UNKNOWN DOMAINS)
    # -----------------------------
    try:
        X_char = char_vectorizer.transform([domain])
        X_num = url_features(pd.Series([domain]))
        X_final = hstack([X_char, X_num])

        probs = model.predict_proba(X_final)
        phishing_index = list(model.classes_).index(1)
        phishing_prob = float(probs[0, phishing_index])

        return {
            "url": url,
            "domain": domain,
            "prediction": "phishing"
            if phishing_prob >= PHISHING_THRESHOLD
            else "benign",
            "phishing_probability": round(phishing_prob, 4)
        }

    except Exception as e:
        raise HTTPException(500, f"Prediction error: {str(e)}")

# -----------------------------
# LOCAL DEV
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
