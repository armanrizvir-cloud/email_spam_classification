import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import os
import json

try:
    from src.preprocessor import SpamPreprocessor
    from src.models import SpamModelTrainer
except ModuleNotFoundError:  # flat layout: files next to this script
    from preprocessor import SpamPreprocessor
    from models import SpamModelTrainer

def run_training_pipeline():
    print("🚀 Starting MailGuard Training Pipeline...\n")
    
    # 1. Load Data
    data_path = 'data/emails.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run download_data.py first.")
    
    df = pd.read_csv(data_path)
    
    # Standardize column names if needed. HuggingFace dataset usually has 'text' and 'label'
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")
    
    # Map labels to 0 and 1
    if df['label'].dtype == object:
        df['label'] = df['label'].map({'spam': 1, 'not_spam': 0})
        
    print(f"Data loaded successfully. Total records: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts(normalize=True) * 100}\n")
    
    # 2. Train/Test Split
    # We use stratified split to maintain the exact spam/ham ratio in both sets
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    print(f"Training set size: {len(X_train_raw)}")
    print(f"Test set size: {len(X_test_raw)}\n")
    
    # 3. Preprocessing (NLP TF-IDF)
    print("Initializing NLP Preprocessor...")
    preprocessor = SpamPreprocessor(max_features=5000)
    
    # Handle missing values in text
    X_train_clean = X_train_raw.fillna('')
    X_test_clean = X_test_raw.fillna('')
    
    print("Vectorizing text data (this may take a moment)...")
    X_train_vec = preprocessor.fit_transform(X_train_clean)
    X_test_vec = preprocessor.transform(X_test_clean)
    
    # 4. Model Training & Evaluation
    trainer = SpamModelTrainer()
    results = trainer.train_and_evaluate(X_train_vec, y_train, X_test_vec, y_test)
    
    # 5. Intelligent Model Selection
    best_name, best_model = trainer.select_best_model()
    
    # Save metrics to JSON for the dashboard
    metrics_to_save = {}
    for name, res in results.items():
        metrics_to_save[name] = {
            "Accuracy": res["Accuracy"],
            "Precision": res["Precision"],
            "Recall": res["Recall"],
            "F1-Score": res["F1_Score"],
            "Train_Time_sec": res["Train_Time_sec"]
        }
        
    os.makedirs('models', exist_ok=True)
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
        
    # 6. Save the Best Model and Vectorizer for the Dashboard
    print(f"\nSaving {best_name} and Vectorizer to disk...")
    preprocessor.save_vectorizer('models/tfidf_vectorizer.pkl')
    joblib.dump(best_model, 'models/best_model.pkl')
    
    print("\n✅ Training Pipeline Completed Successfully! You can now run the Dashboard.")

if __name__ == "__main__":
    run_training_pipeline()
