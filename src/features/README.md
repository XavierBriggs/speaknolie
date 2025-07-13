# Feature Extraction (`src/features/`)

This directory contains modules for extracting text features from audio transcripts for deepfake detection.

## Files

### `tfidf_features.py`
**Purpose**: Extract TF-IDF (Term Frequency-Inverse Document Frequency) features from transcripts

**Features**:
- Converts text transcripts to sparse feature vectors
- Supports unigrams and bigrams
- Configurable feature parameters
- Automatic data splitting
- Vectorizer saving/loading

**Parameters**:
- `max_features`: 2000 (default)
- `ngram_range`: (1, 2) - unigrams and bigrams
- `min_df`: 2 - minimum document frequency
- `max_df`: 0.95 - maximum document frequency
- `stop_words`: 'english' - remove common words

**Usage**:
```bash
# Basic usage
python src/features/tfidf_features.py --input data/processed/all_transcripts_large.csv

# With custom parameters
python src/features/tfidf_features.py --input data/processed/all_transcripts_large.csv \
    --max_features 3000 --ngram_range "1,3" --split
```

**Output**:
- `tfidf_features.pkl`: Feature matrix and labels
- `tfidf_vectorizer.pkl`: Fitted TF-IDF vectorizer

### `bert_features.py`
**Purpose**: Extract BERT embeddings from transcripts

**Features**:
- Uses BERT-base-uncased model
- Processes in batches for efficiency
- GPU acceleration support
- 768-dimensional embeddings
- Semantic understanding of text

**Parameters**:
- `model_name`: 'bert-base-uncased' (default)
- `max_length`: 512 - maximum sequence length
- `batch_size`: 32 - processing batch size
- `save_model`: True - save model components

**Usage**:
```bash
# Basic usage
python src/features/bert_features.py --input data/processed/all_transcripts_large.csv

# With custom parameters
python src/features/bert_features.py --input data/processed/all_transcripts_large.csv \
    --batch_size 64 --max_length 256 --split
```

**Output**:
- `bert_features.pkl`: Feature matrix and labels
- `bert_model.pkl`: Model components (tokenizer, config)

## Feature Comparison

| Feature Type | Dimensions | Processing Speed | Semantic Understanding | Memory Usage |
|--------------|------------|------------------|----------------------|--------------|
| **TF-IDF** | 2,000 | Fast | Lexical patterns | Low |
| **BERT** | 768 | Slow | Semantic patterns | High |

## Technical Details

### TF-IDF Features
- **Algorithm**: Term Frequency-Inverse Document Frequency
- **Vocabulary**: 2,000 most frequent terms
- **N-grams**: Unigrams + bigrams
- **Preprocessing**: Lowercase, stop words removal
- **Sparsity**: ~95% sparse matrix

### BERT Features
- **Model**: BERT-base-uncased (110M parameters)
- **Tokenization**: WordPiece tokenization
- **Embeddings**: [CLS] token embeddings (768D)
- **Context**: Bidirectional attention
- **Preprocessing**: Automatic tokenization

## Data Flow

### Input Format
```csv
file_number,label,transcription
0,fake,"Who knows the end?"
1,fake,"Then the sparks played amazingly..."
2,real,"Hello world, this is real speech..."
```

### Output Format
```python
# TF-IDF
X: scipy.sparse.csr_matrix (n_samples, 2000)
y: numpy.ndarray (n_samples,)

# BERT
X: numpy.ndarray (n_samples, 768)
y: numpy.ndarray (n_samples,)
```

## Performance Considerations

### TF-IDF Processing
- **Speed**: Very fast (~seconds for 30K samples)
- **Memory**: Low (sparse matrices)
- **Scalability**: Excellent
- **GPU**: Not needed

### BERT Processing
- **Speed**: Slow (~8 minutes for 30K samples on CPU)
- **Memory**: High (model + embeddings)
- **Scalability**: Limited by memory
- **GPU**: Recommended for large datasets

## Best Practices

### For TF-IDF
1. Start with default parameters
2. Adjust `max_features` based on dataset size
3. Use `min_df` to remove rare terms
4. Consider `max_df` to remove very common terms

### For BERT
1. Use GPU if available
2. Adjust batch size based on memory
3. Consider smaller models for speed
4. Save model components for reuse

### For Both
1. Always use `--split` for train/val/test
2. Save vectorizers/models for inference
3. Monitor processing progress
4. Check feature quality before training

## Error Handling

### Common Issues
1. **Memory errors**: Reduce batch size or max_features
2. **Model download**: Automatic with transformers
3. **CUDA errors**: Falls back to CPU
4. **File not found**: Check input path

### Troubleshooting
```bash
# Check available memory
free -h

# Monitor GPU usage
nvidia-smi

# Test with small dataset first
head -1000 data.csv > test_data.csv
```

## Integration with Training

### Loading Features
```python
import pickle

# Load TF-IDF features
with open('tfidf_features.pkl', 'rb') as f:
    data = pickle.load(f)
X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']

# Load BERT features
with open('bert_features.pkl', 'rb') as f:
    data = pickle.load(f)
X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
```

### Model Training
```bash
# Train with TF-IDF features
python src/models/train_nn.py --features tfidf

# Train with BERT features
python src/models/train_nn.py --features bert
``` 