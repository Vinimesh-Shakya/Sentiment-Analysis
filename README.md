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