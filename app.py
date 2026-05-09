"""
===========================================================================
  Twitter Entity Sentiment Analysis Dashboard
  Built with Streamlit + Hugging Face Transformers (bert-base-uncased)
  Fine-tuned 3-class classifier: Negative | Neutral | Positive
===========================================================================
"""

import re
import torch
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification

# ---------------------------------------------------------------------------
# 0.  PAGE CONFIG  (must be the very first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Twitter Entity Sentiment Analysis",
    page_icon="🐦",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 1.  CUSTOM CSS  – "Antigravity" floating-card aesthetic
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu        { visibility: hidden; }
footer           { visibility: hidden; }
header           { visibility: hidden; }

/* ── Dark gradient background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* ── Main container card ── */
.block-container {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 2.5rem 3rem !important;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1a1a3e 0%, #16213e 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}
[data-testid="stSidebar"] * { color: #e0e0ff !important; }

/* ── Floating metric card ── */
.metric-card {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 15px;
    padding: 1.4rem 1.8rem;
    text-align: center;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(8px);
}
.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.5);
}
.metric-card .label {
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #a0a8d0;
    margin-bottom: 0.4rem;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
}
.metric-card .delta {
    font-size: 0.78rem;
    color: #6ee7b7;
    margin-top: 0.2rem;
}

/* ── Submit button ── */
div.stButton > button,
div.stFormSubmitButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: #ffffff !important;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 2rem;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    cursor: pointer;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    width: 100%;
}
div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 28px rgba(102, 126, 234, 0.65);
}

/* ── Text area ── */
textarea {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    color: #e8e8ff !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.25) !important;
}

/* ── Selectbox ── */
div[data-baseweb="select"] > div {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    color: #e8e8ff !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    transition: transform 0.3s ease;
}
[data-testid="stExpander"]:hover { transform: translateY(-2px); }

/* ── Labels & general text ── */
label, .stTextArea label, .stSelectbox label, p {
    color: #c8ceff !important;
}
h1, h2, h3 { color: #ffffff !important; }

/* ── Info / success boxes ── */
.stAlert { border-radius: 12px !important; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.1); }

/* ── Spinner text ── */
.stSpinner > div { color: #667eea !important; }

/* ── Pill badge ── */
.pill {
    display: inline-block;
    padding: 0.2rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.06em;
}
.pill-positive { background: rgba(52,211,153,0.18); color: #34d399; border: 1px solid #34d399; }
.pill-neutral  { background: rgba(148,163,184,0.18); color: #94a3b8; border: 1px solid #94a3b8; }
.pill-negative { background: rgba(248,113,113,0.18); color: #f87171; border: 1px solid #f87171; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 2.  CONSTANTS
# ---------------------------------------------------------------------------

MODEL_DIR   = "./bert_sentiment"       # path to saved fine-tuned model
MAX_LEN     = 128                      # max tokenizer sequence length
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label map: model output index → display string
LABEL_MAP = {
    0: "Negative 🔴",
    1: "Neutral ⚪",
    2: "Positive 🟢",
}

# Pill CSS classes for each sentiment
PILL_CLASS = {
    0: "pill pill-negative",
    1: "pill pill-neutral",
    2: "pill pill-positive",
}

# Entities available in the selectbox
ENTITIES = [
    "Microsoft",
    "Borderlands",
    "FIFA",
    "Apple",
    "Google",
    "Tesla",
    "Amazon",
    "Twitter",
    "PlayStation",
    "Xbox",
    "Other",
]

# ---------------------------------------------------------------------------
# 3.  MODEL LOADING  (cached so it only runs once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="🔄  Loading BERT model — this only happens once…")
def load_model():
    """
    Load the fine-tuned BertTokenizer and BertForSequenceClassification
    from the local ./bert_sentiment directory.

    Returns
    -------
    tokenizer : BertTokenizer
    model     : BertForSequenceClassification  (eval mode, on DEVICE)
    """
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model     = BertForSequenceClassification.from_pretrained(
        MODEL_DIR,
        num_labels=3,          # Negative / Neutral / Positive
    )
    model.to(DEVICE)
    model.eval()               # disable dropout for deterministic inference
    return tokenizer, model


# ---------------------------------------------------------------------------
# 4.  TEXT CLEANING
# ---------------------------------------------------------------------------

def clean_tweet(text: str) -> str:
    """
    Preprocess a raw tweet before feeding it into BERT.

    Steps
    -----
    1. Remove URLs            (http/https/www…)
    2. Remove @mentions
    3. Remove #hashtag symbols (keep the word itself)
    4. Remove special characters / punctuation
    5. Collapse extra whitespace
    6. Convert to lowercase
    """
    # 1. Strip URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 2. Strip @mentions
    text = re.sub(r"@\w+", "", text)

    # 3. Strip the '#' symbol but keep the word
    text = re.sub(r"#(\w+)", r"\1", text)

    # 4. Remove everything except letters, digits, and spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # 5. Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # 6. Lowercase
    text = text.lower()

    return text


# ---------------------------------------------------------------------------
# 5.  PREDICTION
# ---------------------------------------------------------------------------

def predict_sentiment(text: str, tokenizer, model):
    """
    Run inference on a single (cleaned) tweet string.

    Parameters
    ----------
    text      : pre-cleaned tweet string
    tokenizer : loaded BertTokenizer
    model     : loaded BertForSequenceClassification in eval mode

    Returns
    -------
    label_idx   : int   (0 = Negative, 1 = Neutral, 2 = Positive)
    label_str   : str   human-readable label with emoji
    confidence  : float confidence percentage (0–100)
    probs       : list[float]  all three class probabilities (%)
    """
    # Tokenise
    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Move tensors to the same device as the model
    input_ids      = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    # Forward pass — no gradient tracking needed during inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits                             # shape: (1, 3)

    # Convert logits → probabilities via Softmax
    softmax_probs = torch.softmax(logits, dim=1).squeeze()  # shape: (3,)
    probs         = (softmax_probs * 100).tolist()           # as percentages

    # Predicted class
    label_idx = int(torch.argmax(softmax_probs).item())
    label_str = LABEL_MAP[label_idx]
    confidence = probs[label_idx]

    return label_idx, label_str, confidence, probs


# ---------------------------------------------------------------------------
# 6.  SIDEBAR  – Model Diagnostics
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🧠 Model Diagnostics")
    st.markdown("---")

    # ── Model specs ──
    st.markdown("### ⚙️ Model Specs")
    specs = {
        "Architecture":      "BERT Base Uncased",
        "Parameters":        "110 M",
        "Max Sequence":      "128 tokens",
        "Classes":           "3  (Neg / Neu / Pos)",
        "Val Accuracy":      "97.22 %",
        "Inference Device":  str(DEVICE).upper(),
    }
    for key, val in specs.items():
        st.markdown(
            f"<div style='display:flex; justify-content:space-between; "
            f"margin-bottom:6px;'>"
            f"<span style='color:#a0a8d0; font-size:0.82rem;'>{key}</span>"
            f"<span style='color:#e8e8ff; font-weight:600; font-size:0.82rem;'>{val}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Team members ──
    st.markdown("### 👥 Team Members")
    team = [
        ("Mohd. Kaif ", "[MCS25017]"),
        ("Divyansh Gehlot", "[MCS25030]"),
        ("Sandeep Choudhary", "[MCS25034]"),
        ("Vinimesh Shakya", "[MCS25042]"),
        ("Ashish Vishwakarma", "[MCS25053]"),
    ]
    for name, roll in team:
        st.markdown(
            f"<div style='margin-bottom:8px;'>"
            f"<span style='color:#e8e8ff; font-weight:600;'>🎓 {name}</span><br/>"
            f"<span style='color:#a0a8d0; font-size:0.78rem;'>{roll}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#6b7280; font-size:0.75rem;'>"
        "NLP Project · IIIT Lucknow · 2026</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 7.  MAIN PAGE  – Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div style='text-align:center; margin-bottom: 0.5rem;'>
        <h1 style='font-size:2.4rem; font-weight:700; margin-bottom:0;'>
            🐦 Twitter Entity Sentiment Analysis
        </h1>
        <p style='color:#a0a8d0; font-size:1rem; margin-top:0.4rem;'>
            Powered by a fine-tuned <strong>BERT-base-uncased</strong> model
            achieving <strong>97.22 %</strong> validation accuracy.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Brief description
st.markdown(
    """
    > **How it works:** Paste any tweet into the box below, select the entity being discussed,
    > and click **Analyse Sentiment**. The text is first cleaned (URLs, mentions, and special
    > characters removed) and then fed into our fine-tuned BERT model which returns a
    > **Negative / Neutral / Positive** prediction together with a confidence score.
    """,
    unsafe_allow_html=True,
)

st.markdown("<br/>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 8.  LOAD MODEL (triggers the cached function)
# ---------------------------------------------------------------------------

tokenizer, model = load_model()

# ---------------------------------------------------------------------------
# 9.  INPUT FORM
# ---------------------------------------------------------------------------

with st.form(key="sentiment_form", clear_on_submit=False):

    entity = st.selectbox(
        "🏷️  Entity / Brand",
        options=ENTITIES,
        help="Select the entity this tweet is about to provide context.",
    )

    raw_tweet = st.text_area(
        "📝  Raw Tweet",
        placeholder="e.g. @Microsoft's new Surface Pro is absolutely amazing! 🔥 #tech #Microsoft",
        height=130,
        help="Paste the raw tweet text here — URLs, mentions, hashtags are cleaned automatically.",
    )

    submitted = st.form_submit_button("🚀  Analyse Sentiment")

# ---------------------------------------------------------------------------
# 10.  INFERENCE & RESULTS
# ---------------------------------------------------------------------------

if submitted:
    # ── Validation ──
    if not raw_tweet.strip():
        st.warning("⚠️  Please enter a tweet before analysing.", icon="⚠️")
        st.stop()

    # ── Clean text ──
    cleaned = clean_tweet(raw_tweet)

    if not cleaned:
        st.error(
            "❌  After cleaning, the tweet is empty. "
            "Try entering a tweet with actual words.",
            icon="❌",
        )
        st.stop()

    # ── Run inference inside a spinner ──
    with st.spinner("🔍  Running BERT inference…"):
        label_idx, label_str, confidence, probs = predict_sentiment(
            cleaned, tokenizer, model
        )

    st.markdown("---")
    st.markdown("### 📊 Prediction Results")
    st.markdown(
        f"<p style='color:#a0a8d0; margin-top:-0.6rem;'>Entity context: "
        f"<strong style='color:#e8e8ff;'>{entity}</strong></p>",
        unsafe_allow_html=True,
    )

    # ── Side-by-side metric cards ──
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">Predicted Sentiment</div>
                <div class="value">{label_str}</div>
                <div class="delta">Class index: {label_idx}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">Confidence Score</div>
                <div class="value">{confidence:.1f}%</div>
                <div class="delta">Model certainty</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        # Show the runner-up probability
        sorted_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        runner_up_idx, runner_up_prob = sorted_probs[1]
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">Runner-up Class</div>
                <div class="value">{runner_up_prob:.1f}%</div>
                <div class="delta">{LABEL_MAP[runner_up_idx]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── All-class probability bar chart ──
    st.markdown("#### Class Probability Distribution")
    prob_data = {
        "Negative 🔴": round(probs[0], 2),
        "Neutral ⚪":  round(probs[1], 2),
        "Positive 🟢": round(probs[2], 2),
    }

    # Render horizontal progress bars manually for style consistency
    colors = {
        "Negative 🔴": "#f87171",
        "Neutral ⚪":  "#94a3b8",
        "Positive 🟢": "#34d399",
    }
    for cls_name, prob_val in prob_data.items():
        color = colors[cls_name]
        st.markdown(
            f"""
            <div style='margin-bottom:0.6rem;'>
                <div style='display:flex; justify-content:space-between;
                            color:#c8ceff; font-size:0.85rem; margin-bottom:3px;'>
                    <span>{cls_name}</span>
                    <span style='font-weight:600;'>{prob_val:.2f}%</span>
                </div>
                <div style='background:rgba(255,255,255,0.08); border-radius:999px;
                            height:10px; overflow:hidden;'>
                    <div style='width:{prob_val}%; height:100%; border-radius:999px;
                                background:{color}; transition:width 0.6s ease;'></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Expander: cleaned text ──
    with st.expander("🔬  View Preprocessed Input (what BERT actually saw)"):
        st.markdown(
            f"""
            <div style='background:rgba(0,0,0,0.3); border-radius:10px;
                        padding:1rem 1.2rem; color:#e8e8ff;
                        font-family:monospace; font-size:0.9rem;
                        border: 1px solid rgba(255,255,255,0.1);'>
                {cleaned if cleaned else "<em style='color:#6b7280;'>— empty after cleaning —</em>"}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "ℹ️  URLs, @mentions, #symbols, and special characters have been stripped. "
            "Text has been lowercased."
        )

# ---------------------------------------------------------------------------
# 11.  FOOTER TIP (always visible)
# ---------------------------------------------------------------------------

st.markdown("<br/><br/>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align:center; color:#4b5563; font-size:0.78rem;'>
        Built with ❤️ using Streamlit · Hugging Face Transformers · PyTorch<br/>
        IIIT Lucknow · NLP Project 2026
    </div>
    """,
    unsafe_allow_html=True,
)
