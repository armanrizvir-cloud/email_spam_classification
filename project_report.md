# Project Report: MailGuard (Intelligent Email Spam Classification)

## 1. Abstract
The exponential growth of email communication has led to a massive increase in unsolicited and potentially dangerous spam messages. Traditional rule-based filters are no longer sufficient against sophisticated spammers who constantly evolve their tactics. This project, **MailGuard**, presents an intelligent, automated Machine Learning pipeline for email spam classification. By leveraging Natural Language Processing (NLP) techniques—specifically TF-IDF vectorization—unstructured email text is transformed into high-dimensional numerical features. Multiple classification algorithms (Naive Bayes, Logistic Regression, Random Forest, and Linear SVC) were evaluated. The Linear SVC model was intelligently selected for achieving a 99.7% Precision and 99.6% F1-Score, ensuring minimal false positives. The final model is deployed via an interactive Python Dash web application for real-time inference and explainability.

## 2. Introduction
Spam emails are not merely an annoyance; they consume vast amounts of network bandwidth, decrease worker productivity, and frequently serve as vectors for phishing attacks and malware distribution. As a 3rd-year Data Science & AI student, this project aims to transition from theoretical machine learning to a practical, deployable product. Rather than relying on static blacklists, this project applies statistical machine learning to learn the underlying patterns and vocabulary of spam.

## 3. Problem Statement
To design and implement a highly precise text classification system capable of distinguishing between legitimate emails (Ham) and Spam in real-time, while explicitly minimizing False Positives to prevent critical information loss.

## 4. Objectives
1. To clean and preprocess unstructured email data using modern NLP techniques.
2. To extract meaningful numerical features using Term Frequency-Inverse Document Frequency (TF-IDF).
3. To train and rigorously evaluate multiple machine learning algorithms.
4. To implement an automated model selection pipeline based on F1-Score and Precision.
5. To deploy the optimal model in an interactive, user-friendly web dashboard.

## 5. Literature / Background
Historically, spam detection relied on Naive Bayes classifiers because of their simplicity and speed. However, as spammers adapted by injecting "good words" to bypass filters, more robust linear models (like Support Vector Machines and Logistic Regression) became necessary. This project builds upon those foundations, applying modern Python libraries (Scikit-Learn, Plotly) to automate the evaluation process and visually explain the model's decisions.

## 6. Methodology
The project follows a standard Machine Learning lifecycle:
1. **Data Acquisition:** Downloading a labeled corpus from HuggingFace.
2. **Preprocessing:** Handling missing values, standardizing labels (1=Spam, 0=Ham), and lowercasing text.
3. **Vectorization:** Converting text to sparse matrices using TF-IDF (unigrams and bigrams).
4. **Model Training:** Fitting multiple algorithms using a stratified 80/20 train/test split.
5. **Evaluation:** Comparing Precision, Recall, and F1-Scores.
6. **Deployment:** Loading serialized `.pkl` models into a Dash server for inference.

## 7. System Architecture
The system consists of three main components:
* **The NLP Engine (`src/preprocessor.py`):** Responsible for parsing raw strings into arrays of mathematical weights.
* **The Automated Pipeline (`train_pipeline.py`):** Orchestrates the data flow, trains the models, ranks them, and saves the winner to the local disk.
* **The Web Application (`dashboard/app.py`):** A frontend UI that captures user input, requests a prediction from the serialized model, and renders the result and confidence score.

## 8. Dataset Description
The dataset was sourced from HuggingFace (`Deysi/spam-detection-dataset`), containing 10,900 emails.
* **Features:** A single text column containing the raw body of the email.
* **Target:** A binary label (Spam / Not Spam).
* **Distribution:** The dataset is highly balanced, with approximately 50.4% Spam and 49.5% Legitimate emails.

## 9. Data Preprocessing
* Missing values were imputed with empty strings to prevent row deletion.
* The target labels were mapped to binary integers (0 and 1) for compatibility with Scikit-Learn's strict evaluation metrics.
* English stop-words ('the', 'is', 'at') were removed during vectorization to reduce noise.

## 10. Exploratory Data Analysis (EDA)
EDA (implemented in the Dashboard) focuses on word importance. By examining the coefficients assigned by the linear models, we can definitively prove that the model has learned the correct patterns (e.g., assigning high positive weights to words like 'viagra', 'winner', 'click', and negative weights to 'meeting', 'project', 'attached').

## 11. Algorithms Evaluated
1. **Multinomial Naive Bayes:** The standard baseline for text data.
2. **Logistic Regression:** Used for its strong interpretability and probability outputs.
3. **Random Forest:** An ensemble method tested to see if non-linear splits could capture complex sentence structures.
4. **Linear SVC:** An implementation of Support Vector Machines heavily optimized for the sparse datasets created by TF-IDF.

## 12. Model Evaluation
Accuracy was deliberately avoided as the sole metric. 
* **Precision** was prioritized because a False Positive (a legitimate email marked as spam) is catastrophic for the user.
* **F1-Score** was used as the primary tie-breaker to ensure a balance between Precision and Recall.

## 13. Results
The models performed exceptionally well due to the clean nature of the dataset.
* **Logistic Regression:** Accuracy 99.6% | Precision 99.8% | F1 99.6%
* **Linear SVC:** Accuracy 99.6% | Precision 99.7% | F1 99.6%
* *Note: Linear SVC was selected by the automated pipeline based on the programmatic tie-breaker logic.*

## 14. Dashboard
The interactive dashboard was built using Plotly Dash and the Dash-Bootstrap-Components library (CYBORG theme). It features:
* A Live Prediction tab for real-time text analysis.
* A Model Comparison tab with grouped bar charts.
* A Spam Insights tab that reverse-engineers the model's coefficients to show exactly which words triggered the alert.

## 15. Limitations
* **Multilingual Support:** The model is currently trained exclusively on English emails and uses English stop-word removal.
* **Context Loss:** TF-IDF relies on word frequencies and completely ignores the sequential, semantic context of a sentence (unlike Recurrent Neural Networks or Transformers).

## 16. Future Scope
1. Implement a Transformer model (such as BERT) to capture deep semantic meaning and sarcasm.
2. Develop a Google Chrome extension or Outlook plugin that interfaces with the model's API to scan emails directly inside the browser.
3. Incorporate metadata features (e.g., sender email address domain, time of day) alongside the raw text.

## 17. Conclusion
The MailGuard project successfully demonstrates that statistical machine learning combined with basic NLP can achieve near-perfect classification on structured datasets. By emphasizing precision and automating the model selection process, this project avoids common pitfalls in academic ML projects, resulting in a robust, deployable application suitable for real-world scenarios.
