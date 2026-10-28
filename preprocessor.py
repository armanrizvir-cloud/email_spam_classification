import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

class SpamPreprocessor:
    def __init__(self, max_features=5000):
        # We use unigrams and bigrams to catch phrases like "win money"
        self.vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=max_features,
            ngram_range=(1, 2)
        )
        self.is_fitted = False

    def clean_text(self, df):
        """Basic cleaning: handle missing values and lowercase."""
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Fill missing text with empty strings
        df['text'] = df['text'].fillna('')
        
        # Lowercase everything (TF-IDF vectorizer does this by default, but good for safety)
        df['text'] = df['text'].str.lower()
        return df

    def fit_transform(self, X):
        """Fit the vectorizer on the training data and transform it."""
        X_vec = self.vectorizer.fit_transform(X)
        self.is_fitted = True
        return X_vec

    def transform(self, X):
        """Transform new data using the already fitted vectorizer."""
        if not self.is_fitted:
            raise ValueError("The vectorizer is not fitted yet. Call fit_transform first.")
        return self.vectorizer.transform(X)

    def save_vectorizer(self, filepath):
        """Save the fitted vectorizer for the Dashboard inference."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.vectorizer, filepath)
        print(f"Vectorizer saved successfully at: {filepath}")

