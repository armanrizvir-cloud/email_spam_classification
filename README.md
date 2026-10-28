# 📧 MailGuard: Intelligent Email Spam Classification

## Overview
MailGuard is a highly accurate Machine Learning pipeline and interactive web application designed to classify emails as Legitimate (Ham) or Spam using Natural Language Processing (NLP). Built as a 3rd-year B.Tech DS & AI project, it focuses on real-world applicability by prioritizing **Precision** (minimizing false positives to protect legitimate emails).

## Key Features
* **Zero-Leakage ML Pipeline**: Robust separation of text preprocessing, TF-IDF vectorization, and model training.
* **Intelligent Selection**: Automatically trains Naive Bayes, Logistic Regression, Random Forest, and Linear SVC, selecting the best model based on F1-Score and Precision.
* **Explainable AI (XAI)**: Extracts model coefficients to show exactly *which* words triggered the spam filter.
* **Premium Dashboard**: A sleek, dark-themed UI built with Python Dash.

## Tech Stack
* **Language:** Python 3
* **Data Processing:** Pandas, Datasets
* **Machine Learning:** Scikit-Learn (TF-IDF, Linear SVC, etc.)
* **Web Framework:** Dash, Dash Bootstrap Components
* **Visualization:** Plotly Express

## Project Structure
```
project/
├── data/                  # Downloaded dataset (emails.csv)
├── src/                   # Source code modules
│   ├── preprocessor.py    # TF-IDF NLP logic
│   └── models.py          # ML algorithms & selection logic
├── models/                # Saved .pkl files and metrics
├── dashboard/             # Web application
│   └── app.py
├── docs/                  # Generated PPT and documentation
├── download_data.py       # Script to fetch HuggingFace dataset
├── train_pipeline.py      # Automated training pipeline
├── create_ppt.py          # Script to generate the presentation
├── .gitignore
└── README.md
```

## How to Run
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Download Data:**
   ```bash
   python3 download_data.py
   ```
3. **Train the Models:**
   ```bash
   python3 train_pipeline.py
   ```
4. **Launch the Dashboard:**
   ```bash
   python3 dashboard/app.py
   ```
   *Open `http://127.0.0.1:8050/` in your browser.*

## Author
**Arman Rizvi**
*3rd Year B.Tech, Data Science & AI*
