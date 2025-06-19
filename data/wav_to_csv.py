import os
import pandas as pd
import whisper
from tqdm import tqdm
import ssl
import certifi

# Fix SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

def load_whisper_model():
    """Load Whisper model with error handling."""
    try:
        return whisper.load_model("large")
    except Exception as e:
        print(f"Error loading Whisper model: {str(e)}")
        print("Trying to download model with SSL verification disabled...")
        # Try again with SSL verification disabled
        ssl._create_default_https_context = ssl._create_unverified_context
        return whisper.load_model("large")

# Load your Whisper model
model = load_whisper_model()

def transcribe(path: str) -> str:
    """Transcribe one .wav file with Whisper and return the text."""
    try:
        res = model.transcribe(path)
        return res["text"].strip()
    except Exception as e:
        print(f"Error transcribing {path}: {str(e)}")
        return ""

def build_whisper_csv(root_dir: str, out_csv: str, max_files: int = None):
    """
    Walk through ./raw/fake and ./raw/real, transcribe each .wav,
    and save to a single CSV—with a progress bar.
    
    Args:
        root_dir: Directory containing fake and real subdirectories
        out_csv: Path to output CSV file
        max_files: Maximum number of files to process (None for all files)
    """
    # 1) Gather all files
    items = []
    for label in ("fake", "real"):
        folder = os.path.join(root_dir, label)
        if not os.path.exists(folder):
            print(f"Warning: Directory {folder} does not exist")
            continue
        for fn in os.listdir(folder):
            if fn.lower().endswith(".wav"):
                items.append((label, os.path.join(folder, fn)))

    if not items:
        print("No WAV files found in the specified directories")
        return

    # Limit number of files if specified
    if max_files is not None:
        items = items[:max_files]
        print(f"Processing {max_files} files (limited from {len(items)} total files)")
    else:
        print(f"Processing all {len(items)} files")

    # 2) Transcribe with progress
    records = []
    errors = []
    for label, filepath in tqdm(items, desc="Transcribing .wav files"):
        stem = os.path.splitext(os.path.basename(filepath))[0]
        try:
            file_number = int(stem)
        except ValueError:
            file_number = stem

        try:
            text = transcribe(filepath)
            if text:
                records.append({
                    "file_number": file_number,
                    "label": label,
                    "transcription": text
                })
            else:
                errors.append({
                    "file_number": file_number,
                    "label": label,
                    "error": "Empty transcription"
                })
        except Exception as e:
            errors.append({
                "file_number": file_number,
                "label": label,
                "error": str(e)
            })

    # 3) Build DataFrame, sort, and write out
    if records:
        df = pd.DataFrame(records)
        df.sort_values(["label", "file_number"], inplace=True)
        df.to_csv(out_csv, index=False)
        print(f"✅ Wrote {len(df)} rows to {out_csv}")
    else:
        print("No successful transcriptions to write")

    if errors:
        error_csv = os.path.splitext(out_csv)[0] + "_errors.csv"
        pd.DataFrame(errors).to_csv(error_csv, index=False)
        print(f"⚠️ Wrote {len(errors)} error records to {error_csv}")

if __name__ == "__main__":
    # Process only 10 files for testing
    build_whisper_csv(
        root_dir="./raw/inTheWild", 
        out_csv="./processed/all_transcripts_large.csv",
        max_files=5000  # Set to None to process all files
    )