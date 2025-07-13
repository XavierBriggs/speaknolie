# Model Training (`src/models/`)

This directory contains the neural network training pipeline and evaluation tools for the Audio Deepfake Detection project.

## Files

### `train_nn.py`
**Purpose**: Main training script for feedforward neural network

**Features**:
- Supports both TF-IDF and BERT features
- Automatic data loading and preprocessing
- PyTorch neural network training
- Comprehensive evaluation metrics
- Model checkpointing and early stopping

**Model Architecture**:
```python
DeepfakeClassifier(
    input_size: int,           # 2000 (TF-IDF) or 768 (BERT)
    hidden_sizes: [512, 256, 128],
    num_classes: 2,            # real/fake
    dropout: 0.3
)
```

**Usage**:
```bash
# Train with TF-IDF features
python src/models/train_nn.py --features tfidf --epochs 20

# Train with BERT features
python src/models/train_nn.py --features bert --epochs 20

# Custom parameters
python src/models/train_nn.py --features tfidf --epochs 30 --batch_size 64 --lr 0.001
```

**Parameters**:
- `--features`: 'tfidf' or 'bert'
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)

### `utils.py`
**Purpose**: Utility functions for model training and evaluation

**Functions**:
- `compute_metrics()`: Calculate accuracy, precision, recall, F1
- Additional helper functions for model evaluation

### `save_results.py`
**Purpose**: Helper script for saving results with proper naming

**Usage**:
```bash
python src/models/save_results.py tfidf
python src/models/save_results.py bert
```

## Training Pipeline

### 1. Data Loading
```python
# Load pre-extracted features
X_train, X_val, X_test, y_train, y_val, y_test = load_features(feature_path)
```

### 2. Model Initialization
```python
model = DeepfakeClassifier(
    input_size=X_train.shape[1],
    hidden_sizes=[512, 256, 128],
    dropout=0.3
)
```

### 3. Training Loop
- **Optimizer**: Adam with weight decay
- **Loss**: CrossEntropyLoss
- **Scheduler**: ReduceLROnPlateau
- **Early Stopping**: 5 epochs patience
- **Checkpointing**: Save best model

### 4. Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1
- **Confusion Matrix**: Detailed error analysis
- **Results**: Saved to pickle files

## Model Performance

### TF-IDF Model Results
- **Accuracy**: 78.85%
- **F1 Score**: 77.97%
- **Precision**: 78.97%
- **Recall**: 78.85%

### BERT Model Results
- **Accuracy**: 81.51%
- **F1 Score**: 81.30%
- **Precision**: 81.32%
- **Recall**: 81.51%

## Training Configuration

### Hyperparameters
```python
# Model Architecture
hidden_sizes = [512, 256, 128]
dropout = 0.3
num_classes = 2

# Training Parameters
batch_size = 32
learning_rate = 0.001
epochs = 20
weight_decay = 1e-5

# Data Split
train_size = 0.6
val_size = 0.2
test_size = 0.2
```

### Optimization
- **Optimizer**: Adam with weight decay
- **Learning Rate**: 0.001 with scheduling
- **Scheduler**: ReduceLROnPlateau (patience=3)
- **Early Stopping**: 5 epochs patience

### Regularization
- **Dropout**: 0.3 in hidden layers
- **Batch Normalization**: After each hidden layer
- **Weight Decay**: 1e-5 L2 regularization

## Evaluation Metrics

### Primary Metrics
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

### Confusion Matrix
```
                Predicted
              Real    Fake
Actual Real   TN      FP
      Fake    FN      TP
```

### Error Analysis
- **False Positives**: Real speech classified as fake
- **False Negatives**: Fake speech classified as real
- **Error Rates**: Per-class error analysis

## Output Files

### Training Results
- `training_results.pkl`: Complete evaluation results
- `best_model.pth`: Trained model weights
- `training_results_tfidf.pkl`: TF-IDF specific results
- `training_results_bert.pkl`: BERT specific results

### Model Files
- `best_model.pth`: PyTorch model state dict
- Model configuration saved in results

## Training Tips

### For Better Performance
1. **Data Quality**: Ensure clean, balanced dataset
2. **Feature Engineering**: Try different feature parameters
3. **Hyperparameter Tuning**: Experiment with learning rates
4. **Ensemble Methods**: Combine multiple models
5. **Cross-validation**: Use k-fold validation

### For Faster Training
1. **GPU Acceleration**: Use CUDA if available
2. **Batch Size**: Increase if memory allows
3. **Early Stopping**: Prevents overfitting
4. **Model Size**: Reduce hidden layers if needed

### For Reproducibility
1. **Random Seeds**: Fixed seeds for consistency
2. **Checkpointing**: Save best models
3. **Logging**: Track all hyperparameters
4. **Version Control**: Track code and data versions

## Troubleshooting

### Common Issues
1. **Memory Errors**: Reduce batch size
2. **Overfitting**: Increase dropout or reduce model size
3. **Underfitting**: Increase model capacity or training time
4. **Slow Training**: Use GPU or reduce model size

### Debugging
```python
# Check data shapes
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")

# Check model parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# Monitor training progress
for epoch in range(epochs):
    train_loss, train_acc = train_epoch(...)
    print(f"Epoch {epoch}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
```

## Integration

### Loading Trained Models
```python
import torch
from train_nn import DeepfakeClassifier

# Load model
model = DeepfakeClassifier(input_size=2000)  # or 768 for BERT
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(features)
```

### Inference Pipeline
```python
# Load features and model
features = load_features('tfidf_features.pkl')
model = load_model('best_model.pth')

# Make predictions
predictions = model.predict(features)
probabilities = model.predict_proba(features)
``` 