"""
bert_features.py
Extracts BERT embeddings from transcript CSVs for deepfake detection.
"""

import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import pickle
import argparse
from tqdm import tqdm

def extract_bert_features(csv_path, model_name="bert-base-uncased", max_length=512, 
                         batch_size=32, save_model=True, model_path=None):
    """
    Loads transcripts from a CSV and computes BERT embeddings.
    
    Args:
        csv_path (str): Path to the CSV file with transcripts.
        model_name (str): Name of the BERT model to use.
        max_length (int): Maximum sequence length for BERT.
        batch_size (int): Batch size for processing.
        save_model (bool): Whether to save the tokenizer and model.
        model_path (str): Path to save/load the model components.
    
    Returns:
        X (array): BERT embedding matrix
        y (array): Labels (real/fake)
        tokenizer: BERT tokenizer
        model: BERT model
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    # Load BERT tokenizer and model
    print(f"Loading BERT model: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    # Set model to evaluation mode
    model.eval()
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Process transcripts in batches
    embeddings = []
    
    print(f"Extracting BERT embeddings with parameters:")
    print(f"  - model: {model_name}")
    print(f"  - max_length: {max_length}")
    print(f"  - batch_size: {batch_size}")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
            batch_texts = df['transcription'].iloc[i:i+batch_size].tolist()
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get BERT outputs
            outputs = model(**inputs)
            
            # Use [CLS] token embeddings (first token)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
    
    # Concatenate all embeddings
    X = np.vstack(embeddings)
    y = df['label'].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Embedding dimension: {X.shape[1]}")
    
    # Save model components if requested
    if save_model and model_path:
        model_components = {
            'tokenizer': tokenizer,
            'model_name': model_name,
            'max_length': max_length
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_components, f)
        print(f"Saved model components to {model_path}")
    
    return X, y, tokenizer, model

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
    """Main function for BERT feature extraction."""
    parser = argparse.ArgumentParser(description="Extract BERT features from transcripts")
    parser.add_argument('--input', type=str, required=True, 
                       help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='bert_features.pkl',
                       help='Path to save extracted features')
    parser.add_argument('--model', type=str, default='bert_model.pkl',
                       help='Path to save/load model components')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                       help='BERT model name')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--split', action='store_true',
                       help='Split data into train/val/test sets')
    
    args = parser.parse_args()
    
    # Extract features
    X, y, tokenizer, model = extract_bert_features(
        csv_path=args.input,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        model_path=args.model
    )
    
    # Split data if requested
    if args.split:
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        
        # Save split data
        split_data_dict = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test
        }
        
        with open(args.output, 'wb') as f:
            pickle.dump(split_data_dict, f)
        print(f"Saved split data to {args.output}")
    else:
        # Save full dataset
        data = {'X': X, 'y': y}
        with open(args.output, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved full dataset to {args.output}")

if __name__ == "__main__":
    main() 