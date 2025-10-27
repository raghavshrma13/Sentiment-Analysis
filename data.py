import pandas as pd
import numpy as np
import os
from typing import Tuple

def load_twitter_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Twitter Entity Sentiment Analysis dataset from separate train/test files
    Dataset columns: ID, Entity, Sentiment, Tweet
    Returns:
        Tuple containing training and test dataframes
    """
    # Define paths
    base_path = os.path.dirname(__file__)
    train_path = os.path.join(base_path, 'twitter_training.csv')
    test_path = os.path.join(base_path, 'twitter_validation.csv')
    
    # Load datasets
    train_df = pd.read_csv(train_path, names=['id', 'entity', 'sentiment', 'text'])
    test_df = pd.read_csv(test_path, names=['id', 'entity', 'sentiment', 'text'])
    
    # Convert sentiment to numerical values
    sentiment_map = {
        'Positive': 1,
        'Negative': -1,
        'Neutral': 0,
        'Irrelevant': 2
    }
    
    # Map sentiments for both datasets
    train_df['sentiment_label'] = train_df['sentiment'].map(sentiment_map)
    test_df['sentiment_label'] = test_df['sentiment'].map(sentiment_map)
    
    # Print dataset information
    print("\nTraining Dataset info:")
    print(f"Shape: {train_df.shape}")
    print("\nTraining Sentiment distribution:")
    print(train_df['sentiment'].value_counts())
    
    print("\nTest Dataset info:")
    print(f"Shape: {test_df.shape}")
    print("\nTest Sentiment distribution:")
    print(test_df['sentiment'].value_counts())
    
    # Create directories
    for dir_path in ['data/raw', 'data/processed', 'data/embeddings']:
        os.makedirs(os.path.join(base_path, dir_path), exist_ok=True)
    
    # Prepare data for C++ preprocessing
    columns_for_cpp = ['id', 'text', 'sentiment_label']
    train_data = train_df[columns_for_cpp].copy()
    test_data = test_df[columns_for_cpp].copy()
    
    # Save in format ready for C++ processing
    train_output = os.path.join(base_path, 'data/raw/train_for_cpp.csv')
    test_output = os.path.join(base_path, 'data/raw/test_for_cpp.csv')
    
    # Save with minimal formatting for easy C++ parsing
    train_data.to_csv(train_output, index=False, quoting=1)
    test_data.to_csv(test_output, index=False, quoting=1)
    
    print(f"\nSaved preprocessed data for C++ at:")
    print(f"- {train_output}")
    print(f"- {test_output}")
    
    return train_data, test_data

if __name__ == "__main__":
    train_df, test_df = load_twitter_data()