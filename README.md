# 🍽️ Restaurant Review Analysis with Transformer-Based Sentiment & Clustering

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![NLP](https://img.shields.io/badge/NLP-Transformers-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

This project provides a full-stack data science pipeline on real-world **Zomato restaurant review data**, combining **NLP with Transformers**, **interactive visualizations**, and **unsupervised learning** to extract hidden insights from customer reviews and metadata.

---

## 🚀 Project Highlights

- 🧠 **Sentiment Analysis** using `distilbert-base-uncased-finetuned-sst-2-english` from Hugging Face Transformers.
- 📊 **Interactive Visualizations** with Plotly to explore rating, cuisine, and sentiment distribution.
- 🤖 **K-Means Clustering** on standardized numerical features (Rating, Review Length, Sentiment Score).
- 📈 **Evaluation** using metrics like Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Score.
- 🔍 **Feature Engineering**: Review length, encoded sentiments, etc.

---

## 📁 Dataset

- `Zomato Restaurant names and Metadata.csv`: Contains metadata such as name, location, cuisine, and rating.
- `Zomato Restaurant reviews.csv`: Contains customer reviews per restaurant.

> 📦 Source: Proprietary dataset provided for analysis (replace this with actual source if public).

---

## 📌 Tech Stack

| Category | Libraries/Tools |
|---------|-----------------|
| Language | Python |
| NLP | Hugging Face Transformers, DistilBERT |
| Visualization | Matplotlib, Seaborn, Plotly |
| Clustering | Scikit-learn (KMeans, metrics) |
| Data Manipulation | Pandas, NumPy |

---

## 🔍 Project Workflow

### 1. Data Preparation
- Load metadata and reviews using `pandas`.
- Merge datasets on restaurant names.
- Identify and handle missing values (mean imputation for ratings).

### 2. Feature Engineering
- Create new features:
  - `ReviewLength` – number of characters in each review
  - `SentimentLabel`, `SentimentScore` – extracted from Transformer pipeline

### 3. Sentiment Analysis with Transformers
```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("Amazing food and ambiance!")
