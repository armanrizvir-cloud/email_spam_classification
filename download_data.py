import pandas as pd
from datasets import load_dataset
import os

def download_and_prepare_dataset():
    print("Downloading Email Spam dataset from HuggingFace...")
    # Load a reliable email spam dataset
    dataset = load_dataset("Deysi/spam-detection-dataset")
    
    # It usually has 'train' and 'test', we can combine them or just use train
    df_train = dataset['train'].to_pandas()
    df_test = dataset['test'].to_pandas()
    
    df = pd.concat([df_train, df_test], ignore_index=True)
    
    # Rename columns if necessary. Typically 'text' and 'label'
    print(f"Dataset columns: {df.columns.tolist()}")
    print(f"Total records downloaded: {len(df)}")
    
    # Save to data/ directory
    os.makedirs('data', exist_ok=True)
    output_path = 'data/emails.csv'
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset successfully saved to {output_path}")

if __name__ == "__main__":
    download_and_prepare_dataset()
