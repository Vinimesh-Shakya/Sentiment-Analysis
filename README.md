# 🐦 Twitter Entity Sentiment Analysis

![Accuracy](https://img.shields.io/badge/Accuracy-97.22%25-brightgreen)
![Model](https://img.shields.io/badge/Model-BERT--base--uncased-blue)
![Framework](https://img.shields.io/badge/Framework-FastAPI%20%7C%20Streamlit-purple)

A real-time sentiment analysis system built for X (formerly Twitter). This project utilizes a fine-tuned **BERT-base-uncased** model to classify tweets about specific entities/brands into three categories: **Negative 🔴, Neutral ⚪, and Positive 🟢**. 

It includes a standalone **Streamlit Dashboard** (featuring an "Antigravity" aesthetic) and a **Chrome Extension** for instant, 1-click on-screen sentiment analysis directly in the browser.

---

## 👥 Team Members
* **Mohd. Kaif** [MCS25017]
* **Divyansh Gehlot** [MCS25030]
* **Sandeep Choudhary** [MCS25034]
* **Vinimesh Shakya** [MCS25042]
* **Ashish Vishwakarma** [MCS25053]

*NLP Project · IIIT Lucknow · 2026*

---

## 🏗️ Project Architecture

```text
twitter-sentiment-project/
├── .gitignore
├── README.md
├── requirements.txt
├── Twitter_Sentiment_analysis.ipynb  # Training and experimentation notebook
├── api.py                            # FastAPI backend server
├── app.py                            # Streamlit dashboard frontend
├── bert_sentiment/                   # Local fine-tuned model weights (Git Ignored)
└── extension/                        # Chrome Extension files
    ├── manifest.json
    ├── content.js
    ├── popup.html
    ├── popup.css
    └── popup.js
```

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/Vinimesh-Shakya/Sentiment-Analysis.git](https://github.com/Vinimesh-Shakya/Sentiment-Analysis.git)
cd Sentiment-Analysis
```

### 2. Install Dependencies
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### 3. Add the Model Weights (Important)
Because of GitHub's file size limits, the fine-tuned model weights are not included in this repository. 
1. Create a folder named `bert_sentiment/` in the root directory.
2. Place your fine-tuned `config.json`, `tokenizer.json`, `tokenizer_config.json`, and `model.safetensors` (or `pytorch_model.bin`) inside this folder.

### 4. Run the FastAPI Backend
The API must be running for the Chrome Extension to work.
```bash
uvicorn api:app --reload
```
The API will be hosted at `http://localhost:8000`.

### 5. Run the Streamlit Dashboard (Optional)
If you want to use the standalone web dashboard:
```bash
streamlit run app.py
```

### 6. Install the Chrome Extension
1. Open Google Chrome and navigate to `chrome://extensions/`.
2. Toggle **Developer mode** in the top right corner.
3. Click **Load unpacked**.
4. Select the `extension/` folder from this project.
5. Open Twitter/X, click the extension icon, and analyze tweets in real-time!

---

## 🛠️ Tech Stack
* **Machine Learning:** PyTorch, Hugging Face Transformers
* **Backend:** Python, FastAPI, Uvicorn
* **Frontend:** Streamlit, HTML/CSS/JavaScript
* **Extension:** Manifest V3, Chrome Extension APIs