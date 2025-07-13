# Data Processing (`src/data/`)

This directory contains scripts for data preparation and processing in the Audio Deepfake Detection project.

## Files

### `prepare_data.py`
**Purpose**: Data quality checking and cleaning for transcript CSVs

**Features**:
- Validates required columns (`file_number`, `label`, `transcription`)
- Checks for missing or empty transcriptions
- Standardizes labels (`real`/`fake` vs `bona-fide`/`spoof`)
- Removes duplicate entries
- Generates data quality reports

**Usage**:
```bash
# Check data quality only
python src/data/prepare_data.py --input all_transcripts_large.csv --check-only

# Clean and save processed data
python src/data/prepare_data.py --input all_transcripts_large.csv --output cleaned_transcripts.csv
```

**Output**:
- Data quality summary
- Cleaned CSV file
- Processing statistics

### `wav_to_csv.py`
**Purpose**: Audio transcription using Whisper model

**Features**:
- Transcribes .wav files to text using Whisper
- Processes both real and fake audio samples
- Handles large datasets with progress bars
- Error handling and logging
- Supports batch processing

**Usage**:
```bash
python data/wav_to_csv.py
```

**Configuration**:
- Model: Whisper "large" model
- Output: CSV with file_number, label, transcription
- Error handling: Separate error log file

## Data Pipeline

### 1. Raw Audio Processing
```
Audio Files (.wav)
    ↓
Whisper Transcription
    ↓
Transcript CSV
```

### 2. Data Cleaning
```
Transcript CSV
    ↓
Quality Check
    ↓
Clean CSV
```

### 3. Feature Extraction
```
Clean CSV
    ↓
TF-IDF/BERT Features
    ↓
Feature Matrix
```

## Data Quality Standards

### Required Columns
- `file_number`: Unique identifier for each audio file
- `label`: Classification (`real` or `fake`)
- `transcription`: Text output from Whisper

### Quality Checks
- ✅ No missing transcriptions
- ✅ No empty transcriptions
- ✅ Consistent label format
- ✅ No duplicate file numbers
- ✅ Valid text content

### Expected Dataset Statistics
- **Total samples**: ~31,700
- **Real samples**: ~19,900 (63%)
- **Fake samples**: ~11,800 (37%)
- **Average transcript length**: Variable

## Error Handling

### Common Issues
1. **Missing files**: Script continues with available files
2. **Transcription errors**: Logged to separate error file
3. **Label inconsistencies**: Automatically standardized
4. **Memory issues**: Batch processing for large datasets

### Error Files
- `all_transcripts_large_errors.csv`: Failed transcriptions
- Processing logs: Detailed error information

## Performance Notes

### Whisper Processing
- **Model size**: Large (1.5GB)
- **Processing speed**: ~1-2 seconds per audio file
- **Memory usage**: High during transcription
- **GPU acceleration**: Available if CUDA is installed

### Data Cleaning
- **Processing speed**: Fast (pandas operations)
- **Memory usage**: Low
- **Scalability**: Handles datasets up to 100K+ samples

## Best Practices

### For Large Datasets
1. Use batch processing for transcription
2. Monitor memory usage
3. Save intermediate results
4. Use progress bars for long operations

### For Quality Assurance
1. Always run data quality checks
2. Review error logs
3. Validate label distributions
4. Check for data leakage

### For Reproducibility
1. Use fixed random seeds
2. Save processing parameters
3. Document data transformations
4. Version control data files 