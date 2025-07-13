"""
train_nn.py
Trains a feedforward neural network for audio deepfake detection using text features.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pickle
import numpy as np
import os
from tqdm import tqdm

class DeepfakeClassifier(nn.Module):
    """Feedforward neural network for deepfake detection."""
    
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], num_classes=2, dropout=0.3):
        super(DeepfakeClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def load_features(feature_path):
    """Load pre-extracted features."""
    with open(feature_path, 'rb') as f:
        data = pickle.load(f)
    
    if 'X_train' in data:  # Split data
        return (data['X_train'], data['X_val'], data['X_test'],
                data['y_train'], data['y_val'], data['y_test'])
    else:  # Full dataset
        return data['X'], data['y']

def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
    """Create PyTorch DataLoaders."""
    
    # Convert labels to integers
    label_map = {'real': 0, 'fake': 1}
    y_train_int = np.array([label_map[y] for y in y_train])
    y_val_int = np.array([label_map[y] for y in y_val])
    y_test_int = np.array([label_map[y] for y in y_test])
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train.toarray() if hasattr(X_train, 'toarray') else X_train)
    X_val_tensor = torch.FloatTensor(X_val.toarray() if hasattr(X_val, 'toarray') else X_val)
    X_test_tensor = torch.FloatTensor(X_test.toarray() if hasattr(X_test, 'toarray') else X_test)
    
    y_train_tensor = torch.LongTensor(y_train_int)
    y_val_tensor = torch.LongTensor(y_val_int)
    y_test_tensor = torch.LongTensor(y_test_int)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, label_map

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in tqdm(train_loader, desc="Training"):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    return total_loss / len(train_loader), correct / total

def evaluate(model, data_loader, criterion, device):
    """Evaluate model on given data."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    return total_loss / len(data_loader), all_predictions, all_labels

def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def main(args):
    """Main training pipeline."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load features
    print(f"Loading features from {args.features}")
    if args.features == 'tfidf':
        feature_path = 'tfidf_features.pkl'
    elif args.features == 'bert':
        feature_path = 'bert_features.pkl'
    else:
        raise ValueError("Features must be 'tfidf' or 'bert'")
    
    if not os.path.exists(feature_path):
        print(f"Error: Feature file {feature_path} not found!")
        print("Please run feature extraction first.")
        return
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_features(feature_path)
    
    print(f"Feature shapes:")
    print(f"  - Train: {X_train.shape}")
    print(f"  - Validation: {X_val.shape}")
    print(f"  - Test: {X_test.shape}")
    
    # Create data loaders
    train_loader, val_loader, test_loader, label_map = create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, 
        batch_size=args.batch_size
    )
    
    # Initialize model
    input_size = X_train.shape[1]
    model = DeepfakeClassifier(
        input_size=input_size,
        hidden_sizes=[512, 256, 128],
        dropout=0.3
    ).to(device)
    
    print(f"Model architecture:")
    print(f"  - Input size: {input_size}")
    print(f"  - Hidden layers: [512, 256, 128]")
    print(f"  - Output size: 2")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    # Training loop
    best_val_f1 = 0
    patience_counter = 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        val_metrics = compute_metrics(val_labels, val_preds)
        val_acc = val_metrics['accuracy']
        val_f1 = val_metrics['f1']
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= 5:
            print("Early stopping triggered!")
            break
    
    # Load best model and evaluate on test set
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load('best_model.pth'))
    
    test_loss, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    test_metrics = compute_metrics(test_labels, test_preds)
    
    print(f"\n=== Final Test Results ===")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"Confusion Matrix:")
    print(test_metrics['confusion_matrix'])
    
    # Save results
    results = {
        'test_metrics': test_metrics,
        'model_config': {
            'input_size': input_size,
            'hidden_sizes': [512, 256, 128],
            'dropout': 0.3
        },
        'label_map': label_map
    }
    
    with open('training_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n✅ Training complete!")
    print(f"Results saved to training_results.pkl")
    print(f"Best model saved to best_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NN for audio deepfake detection")
    parser.add_argument('--features', type=str, default='tfidf', 
                       choices=['tfidf', 'bert'], help='Feature type: tfidf or bert')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    args = parser.parse_args()
    main(args) 