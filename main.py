# ======================================
# PHISHING URL DETECTION - API
# ======================================

import os
import re
import pickle
import hashlib
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# LOAD ARTIFACTS
# -----------------------------
# Load models at startup to avoid reloading on every request
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(BASE_DIR, "phishing_cnb_model.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(BASE_DIR, "char_tfidf_vectorizer.pkl"), "rb") as f:
        char_vectorizer = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Model files not found in {BASE_DIR}. Please ensure 'phishing_cnb_model.pkl' and 'char_tfidf_vectorizer.pkl' are in the correct directory.")
    model = None
    char_vectorizer = None

# -----------------------------
# PREPROCESSING
# -----------------------------
def clean_url(url: str) -> str:
    url = url.lower()
    url = re.sub(r"https?://", "", url)
    url = re.sub(r"www\.", "", url)
    return url.strip()

def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % 1000

def url_features(urls: pd.Series) -> csr_matrix:
    length = urls.str.len().values.reshape(-1, 1)

    digits = urls.apply(
        lambda x: sum(c.isdigit() for c in x)
    ).values.reshape(-1, 1)

    special_chars = urls.apply(
        lambda x: sum(not c.isalnum() for c in x)
    ).values.reshape(-1, 1)

    def extract_tld(url):
        try:
            url = ''.join(c for c in url if ord(c) < 128)
            # Prepend https to ensure urlparse works correctly if scheme is missing
            if not url.startswith("http"):
                 parsed = urlparse("http://" + url)
            else:
                 parsed = urlparse(url)
            
            tld = parsed.netloc.split('.')[-1]
            return stable_hash(tld)
        except Exception:
            return 0

    tld = urls.apply(extract_tld).values.reshape(-1, 1)

    features = np.hstack(
        [length, digits, special_chars, tld]
    ).astype(np.float32)

    return csr_matrix(features)

# -----------------------------
# API MODELS & ENDPOINTS
# -----------------------------
class URLRequest(BaseModel):
    url: str

@app.get("/")
def read_root():
    return {"message": "Phishing URL Guard API is running"}

@app.post("/predict")
def predict_url(request: URLRequest):
    if model is None or char_vectorizer is None:
        raise HTTPException(status_code=500, detail="Model files not loaded properly.")
    
    url = request.url
    
    # Create DataFrame for processing
    df = pd.DataFrame({"url": [url]})
    df["url_clean"] = df["url"].apply(clean_url)

    # TF-IDF features
    try:
        X_char = char_vectorizer.transform(df["url_clean"])
        
        # Numeric features
        X_num = url_features(df["url_clean"])
        
        # Combine features
        X_final = hstack([X_char, X_num])
        
        # Predictions
        preds = model.predict(X_final)
        probs = model.predict_proba(X_final)
        
        pos_index = list(model.classes_).index(1)
        phishing_prob = probs[0, pos_index]
        label = preds[0]
        
        result = {
            "url": url,
            "prediction": "phishing" if label == 1 else "benign",
            "phishing_probability": round(float(phishing_prob), 4)
        }
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# -----------------------------
# LOCAL DEV RUNNER
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
