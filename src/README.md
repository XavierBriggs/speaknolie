# Source Code (`src/`)

This directory contains the main source code for the Audio Deepfake Detection project.

## Directory Structure

```
src/
├── data/           # Data processing scripts
├── features/       # Feature extraction modules
├── models/         # Model training and evaluation
├── visualization/  # Plotting and analysis tools
└── README.md      # This file
```

## Modules Overview

### `data/` - Data Processing
- **`prepare_data.py`**: Data quality checking and cleaning for transcript CSVs
- **`wav_to_csv.py`**: Audio transcription using Whisper (moved from `data/`)

### `features/` - Feature Extraction
- **`tfidf_features.py`**: TF-IDF feature extraction from transcripts
- **`bert_features.py`**: BERT embedding extraction from transcripts

### `models/` - Model Training
- **`train_nn.py`**: Neural network training pipeline
- **`utils.py`**: Utility functions for model training
- **`save_results.py`**: Helper script for saving results with proper naming

### `visualization/` - Analysis and Plotting
- **`plot_results.py`**: Core visualization functions
- **`create_all_plots.py`**: Comprehensive plotting script
- **`generate_report.py`**: Project report generation

## Usage Examples

### Data Preparation
```bash
python src/data/prepare_data.py --input data/processed/all_transcripts_large.csv
```

### Feature Extraction
```bash
# TF-IDF features
python src/features/tfidf_features.py --input data/processed/all_transcripts_large.csv --split

# BERT features
python src/features/bert_features.py --input data/processed/all_transcripts_large.csv --split
```

### Model Training
```bash
# Train with TF-IDF features
python src/models/train_nn.py --features tfidf --epochs 20

# Train with BERT features
python src/models/train_nn.py --features bert --epochs 20
```

### Visualization
```bash
# Create all visualizations
python src/visualization/create_all_plots.py

# Generate project report
python src/visualization/generate_report.py
```

## Key Components

### Feature Extraction Pipeline
1. **TF-IDF**: Extracts 2,000 lexical features (unigrams + bigrams)
2. **BERT**: Extracts 768-dimensional semantic embeddings
3. **Data Splitting**: Automatic train/validation/test split (60/20/20)

### Model Architecture
- **Neural Network**: Feedforward network with [512, 256, 128] hidden layers
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout (0.3) and batch normalization
- **Early Stopping**: Prevents overfitting

### Evaluation Metrics
- Accuracy, Precision, Recall, F1 Score
- Confusion matrix analysis
- Error type analysis (false positives/negatives)

## Dependencies

All required dependencies are listed in `requirements.txt`:
- PyTorch for neural networks
- Transformers for BERT embeddings
- Scikit-learn for TF-IDF and metrics
- Matplotlib/Seaborn for visualization
- Pandas for data manipulation
- Whisper for audio transcription

## File Naming Convention

- `training_results_<model_type>.pkl`: Model evaluation results
- `best_model.pth`: Trained neural network weights
- `tfidf_features.pkl` / `bert_features.pkl`: Extracted features
- `tfidf_vectorizer.pkl` / `bert_model.pkl`: Feature extraction models 