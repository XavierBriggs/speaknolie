# Data Directory (`data/`)

This directory contains data files and processing scripts for the Audio Deepfake Detection project.

## Directory Structure

```
data/
├── processed/           # Processed transcript files
├── raw/               # Raw audio files (if available)
├── external/          # External data sources
├── wav_to_csv.py     # Audio transcription script
└── README.md         # This file
```

## Files

### `wav_to_csv.py`
**Purpose**: Transcribe audio files to text using Whisper

**Features**:
- Processes .wav files from In-the-Wild Audio Deepfake Dataset
- Uses Whisper "large" model for transcription
- Handles both real and fake audio samples
- Progress tracking and error handling
- Outputs CSV with file_number, label, transcription

**Configuration**:
- **Model**: Whisper "large" (1.5GB)
- **Output**: CSV format
- **Error handling**: Separate error log
- **Processing**: Batch processing with progress bars

**Usage**:
```bash
# Process all audio files
python data/wav_to_csv.py

# Configuration in script
root_dir = "./raw/inTheWild"
out_csv = "./processed/all_transcripts_large.csv"
max_files = 5000  # Limit for testing
```

### `processed/`
**Purpose**: Store processed transcript files

**Files**:
- `all_transcripts_large.csv`: Main transcript dataset (31,699 samples)
- `all_transcripts_large_clean.csv`: Cleaned version (if generated)
- `all_transcripts_large_errors.csv`: Failed transcriptions

**Format**:
```csv
file_number,label,transcription
0,fake,"Who knows the end?"
1,fake,"Then the sparks played amazingly..."
2,real,"Hello world, this is real speech..."
```

### `raw/`
**Purpose**: Store raw audio files (if available)

**Structure**:
```
raw/
├── inTheWild/
│   ├── fake/          # AI-generated audio samples
│   └── real/          # Authentic audio samples
```

**File format**: .wav audio files

## Dataset Information

### In-the-Wild Audio Deepfake Dataset
- **Source**: Kaggle (Mohamed, 2023)
- **Size**: 31,779 audio files
- **Distribution**: 63% real, 37% fake
- **Format**: .wav audio files
- **Duration**: Variable length audio clips

### Data Statistics
- **Total samples**: 31,699 (after processing)
- **Real samples**: 19,903 (62.8%)
- **Fake samples**: 11,796 (37.2%)
- **Average transcript length**: Variable
- **Language**: English

### Data Quality
- **Missing transcriptions**: 0
- **Empty transcriptions**: 0
- **Duplicate files**: 0
- **Label consistency**: 100%

## Data Processing Pipeline

### 1. Audio Transcription
```
Raw Audio (.wav)
    ↓
Whisper Model
    ↓
Text Transcripts
    ↓
CSV Output
```

### 2. Data Cleaning
```
Raw Transcripts
    ↓
Quality Check
    ↓
Label Standardization
    ↓
Clean Dataset
```

### 3. Feature Extraction
```
Clean Transcripts
    ↓
TF-IDF Vectorization
    ↓
Feature Matrix
```

## File Naming Convention

### Input Files
- `raw/inTheWild/fake/*.wav`: AI-generated audio
- `raw/inTheWild/real/*.wav`: Authentic audio

### Output Files
- `all_transcripts_large.csv`: Main dataset
- `all_transcripts_large_clean.csv`: Cleaned version
- `all_transcripts_large_errors.csv`: Error log

### Processing Files
- `tfidf_features.pkl`: TF-IDF feature matrix
- `bert_features.pkl`: BERT feature matrix
- `tfidf_vectorizer.pkl`: TF-IDF vectorizer
- `bert_model.pkl`: BERT model components

## Data Validation

### Quality Checks
1. **Required columns**: file_number, label, transcription
2. **Label values**: real, fake (standardized)
3. **No missing data**: All fields populated
4. **No duplicates**: Unique file numbers
5. **Valid text**: Non-empty transcriptions

### Validation Script
```bash
# Check data quality
python src/data/prepare_data.py --input data/processed/all_transcripts_large.csv --check-only

# Clean data
python src/data/prepare_data.py --input data/processed/all_transcripts_large.csv --output data/processed/cleaned.csv
```

## Performance Considerations

### Whisper Processing
- **Model size**: 1.5GB (large model)
- **Processing speed**: ~1-2 seconds per file
- **Memory usage**: High during transcription
- **GPU acceleration**: Available with CUDA

### Storage Requirements
- **Raw audio**: ~10GB (estimated)
- **Transcripts**: ~50MB (CSV)
- **Features**: ~100MB (TF-IDF), ~200MB (BERT)
- **Models**: ~2GB (Whisper + BERT)

## Error Handling

### Common Issues
1. **Missing audio files**: Script continues with available files
2. **Transcription failures**: Logged to error file
3. **Memory errors**: Process in smaller batches
4. **SSL certificate issues**: Handled in script

### Error Files
- `all_transcripts_large_errors.csv`: Failed transcriptions
- Processing logs: Detailed error information

## Best Practices

### For Large Datasets
1. **Batch processing**: Process files in chunks
2. **Progress tracking**: Use progress bars
3. **Error logging**: Save failed transcriptions
4. **Memory management**: Monitor system resources

### For Data Quality
1. **Validation**: Always check data quality
2. **Cleaning**: Remove problematic samples
3. **Standardization**: Consistent label format
4. **Documentation**: Track data transformations

### For Reproducibility
1. **Version control**: Track data files
2. **Processing logs**: Save transformation steps
3. **Random seeds**: Fixed for consistent splits
4. **Backup**: Keep original data safe

## Integration with ML Pipeline

### Feature Extraction
```bash
# Extract TF-IDF features
python src/features/tfidf_features.py --input data/processed/all_transcripts_large.csv --split

# Extract BERT features
python src/features/bert_features.py --input data/processed/all_transcripts_large.csv --split
```

### Model Training
```bash
# Train with processed features
python src/models/train_nn.py --features tfidf
python src/models/train_nn.py --features bert
```

## Troubleshooting

### Common Issues
1. **File not found**: Check directory structure
2. **Memory errors**: Reduce batch size
3. **SSL errors**: Update certificates
4. **Permission errors**: Check file permissions

### Debugging
```bash
# Check file structure
ls -la data/

# Test with small sample
head -100 all_transcripts_large.csv > test_sample.csv

# Verify data quality
python src/data/prepare_data.py --input test_sample.csv --check-only
``` 