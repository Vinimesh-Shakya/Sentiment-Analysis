"""
api.py  —  FastAPI backend for the Chrome Extension
====================================================
Wraps the fine-tuned BERT sentiment model with a single /predict endpoint.
The Chrome extension POSTs tweet text here and gets back a sentiment label
and confidence score.

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import re
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DIR = "./bert_sentiment"
MAX_LEN   = 128
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP = {
    0: "Negative 🔴",
    1: "Neutral ⚪",
    2: "Positive 🟢",
}

# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Twitter Sentiment API",
    description="BERT-based 3-class tweet sentiment classifier.",
    version="1.0.0",
)

# Allow requests from the Chrome extension (chrome-extension://* origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Chrome extensions don't send a standard origin
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  (runs once at startup)
# ─────────────────────────────────────────────────────────────────────────────

print(f"[api] Loading model from {MODEL_DIR} onto {DEVICE} …")
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model     = BertForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=3)
model.to(DEVICE)
model.eval()
print("[api] Model ready ✓")

# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING  (same logic as app.py)
# ─────────────────────────────────────────────────────────────────────────────

def clean_tweet(text: str) -> str:
    """Strip URLs, @mentions, #symbols, special chars; lowercase."""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)   # URLs
    text = re.sub(r"@\w+", "", text)                     # mentions
    text = re.sub(r"#(\w+)", r"\1", text)                # hashtags → word
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)           # special chars
    text = re.sub(r"\s+", " ", text).strip()             # extra spaces
    return text.lower()

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label_idx:  int
    label_str:  str
    confidence: float          # percentage  0–100
    probs:      list[float]    # [neg%, neu%, pos%]
    cleaned:    str            # the preprocessed text that BERT saw

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def health():
    """Quick health-check."""
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(payload: PredictRequest):
    """
    Predict sentiment for a single tweet string.

    Request body:
        { "text": "raw tweet text" }

    Response:
        {
          "label_idx": 2,
          "label_str": "Positive 🟢",
          "confidence": 98.7,
          "probs": [0.5, 0.8, 98.7],
          "cleaned": "preprocessed tweet text"
        }
    """
    raw = payload.text.strip()
    if not raw:
        raise HTTPException(status_code=422, detail="text field is empty.")

    # Clean
    cleaned = clean_tweet(raw)
    if not cleaned:
        raise HTTPException(
            status_code=422,
            detail="Tweet is empty after cleaning. Provide a tweet with actual words."
        )

    # Tokenise
    encoding = tokenizer(
        cleaned,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    # Inference
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    softmax_probs = torch.softmax(logits, dim=1).squeeze()
    probs_list    = (softmax_probs * 100).tolist()   # convert to %

    label_idx  = int(torch.argmax(softmax_probs).item())
    label_str  = LABEL_MAP[label_idx]
    confidence = float(probs_list[label_idx])

    return PredictResponse(
        label_idx=label_idx,
        label_str=label_str,
        confidence=round(confidence, 2),
        probs=[round(p, 2) for p in probs_list],
        cleaned=cleaned,
    )
