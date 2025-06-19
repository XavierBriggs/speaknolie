# Data Directory

This directory contains all data used in the project, organized as follows:

- `raw/` — Raw, unprocessed data (e.g., original .wav files)
- `processed/` — Data that has been cleaned or transformed
- `external/` — Data from external sources

## Using `wav_to_csv.py`

The `wav_to_csv.py` script converts .wav audio files into a CSV format for easier processing in machine learning workflows.

### How to Use

1. **Place your .wav files**
   - Put your .wav files in the appropriate subdirectory under `data/raw/` (e.g., `data/raw/inthewild/`).

2. **Run the script**
   - From the project root, run:
     ```bash
     python data/wav_to_csv.py --input_dir data/raw/inthewild --output_csv data/processed/audio_data.csv
     ```
   - Replace `data/raw/audio` with the path to your .wav files, and `data/processed/audio_data.csv` with your desired output CSV path.

   -Option 2: run it from inside the data dir, just need to edit paths to the data folder 

3. **Script Arguments**
   - `--input_dir`: Directory containing .wav files
   - `--output_csv`: Path to save the resulting CSV file
   - (Optional) Check the script for additional arguments or options

4. **Output**
   - The script will generate a CSV file with features extracted from each .wav file, ready for further processing.

---

## Directory Structure

```
data/
├── raw/         # Raw, original data (e.g., .wav files)
├── processed/   # Processed/cleaned data (e.g., CSVs)
├── external/    # Data from external sources
├── wav_to_csv.py
└── README.md    # This file
```

---

**Tip:** Keep raw data unchanged. All processing should output to `processed/` to preserve original files. 