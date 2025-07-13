"""
tfidf_features.py
Extracts TF-IDF features from transcript CSVs for deepfake detection.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import os
import argparse

def extract_tfidf_features(csv_path, max_features=2000, ngram_range=(1, 2), 
                          min_df=2, max_df=0.95, save_vectorizer=True, 
                          vectorizer_path=None):
    """
    Loads transcripts from a CSV and computes TF-IDF features.
    
    Args:
        csv_path (str): Path to the CSV file with transcripts.
        max_features (int): Maximum number of TF-IDF features.
        ngram_range (tuple): Range of n-grams to consider (e.g., (1,2) for unigrams and bigrams).
        min_df (int): Minimum document frequency for a term to be included.
        max_df (float): Maximum document frequency for a term to be included.
        save_vectorizer (bool): Whether to save the fitted vectorizer.
        vectorizer_path (str): Path to save/load the vectorizer.
    
    Returns:
        X (sparse matrix): TF-IDF feature matrix
        y (array): Labels (real/fake)
        vectorizer: Fitted TfidfVectorizer object
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words='english',  # Remove common English words
        lowercase=True,
        strip_accents='unicode'
    )
    
    print(f"Extracting TF-IDF features with parameters:")
    print(f"  - max_features: {max_features}")
    print(f"  - ngram_range: {ngram_range}")
    print(f"  - min_df: {min_df}")
    print(f"  - max_df: {max_df}")
    
    # Fit and transform the transcriptions
    X = vectorizer.fit_transform(df['transcription'])
    y = df['label'].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of unique features: {len(vectorizer.get_feature_names_out())}")
    
    # Save vectorizer if requested
    if save_vectorizer and vectorizer_path:
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        print(f"Saved vectorizer to {vectorizer_path}")
    
    return X, y, vectorizer

def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split data into train/validation/test sets.
    
    Args:
        X: Feature matrix
        y: Labels
        test_size (float): Proportion for test set
        val_size (float): Proportion for validation set (from remaining data)
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: separate validation set from remaining data
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=random_state, stratify=y_temp
    )
    
    print(f"Data split:")
    print(f"  - Train: {X_train.shape[0]} samples")
    print(f"  - Validation: {X_val.shape[0]} samples") 
    print(f"  - Test: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    """Main function for TF-IDF feature extraction."""
    parser = argparse.ArgumentParser(description="Extract TF-IDF features from transcripts")
    parser.add_argument('--input', type=str, required=True, 
                       help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='tfidf_features.pkl',
                       help='Path to save extracted features')
    parser.add_argument('--vectorizer', type=str, default='tfidf_vectorizer.pkl',
                       help='Path to save/load vectorizer')
    parser.add_argument('--max_features', type=int, default=2000,
                       help='Maximum number of TF-IDF features')
    parser.add_argument('--ngram_range', type=str, default='1,2',
                       help='N-gram range (e.g., "1,2" for unigrams and bigrams)')
    parser.add_argument('--min_df', type=int, default=2,
                       help='Minimum document frequency')
    parser.add_argument('--max_df', type=float, default=0.95,
                       help='Maximum document frequency')
    parser.add_argument('--split', action='store_true',
                       help='Split data into train/val/test sets')
    
    args = parser.parse_args()
    
    # Parse ngram_range
    ngram_range = tuple(map(int, args.ngram_range.split(',')))
    
    # Extract features
    X, y, vectorizer = extract_tfidf_features(
        csv_path=args.input,
        max_features=args.max_features,
        ngram_range=ngram_range,
        min_df=args.min_df,
        max_df=args.max_df,
        vectorizer_path=args.vectorizer
    )
    
    # Split data if requested
    if args.split:
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        
        # Save split data
        import pickle
        split_data_dict = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test
        }
        
        with open(args.output, 'wb') as f:
            pickle.dump(split_data_dict, f)
        print(f"Saved split data to {args.output}")
    else:
        # Save full dataset
        import pickle
        data = {'X': X, 'y': y}
        with open(args.output, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved full dataset to {args.output}")

if __name__ == "__main__":
    main() 