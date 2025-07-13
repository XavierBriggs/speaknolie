"""
prepare_data.py
Data preparation script for audio deepfake detection project.
Checks and cleans transcript CSV data for feature extraction and modeling.
"""

import pandas as pd
import os
import argparse

def check_data_quality(df):
    """
    Check data quality and print summary statistics.
    
    Args:
        df: DataFrame containing transcript data
    
    Returns:
        dict: Summary of data quality issues
    """
    issues = {}
    
    # Check for required columns
    required_cols = ['file_number', 'label', 'transcription']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues['missing_columns'] = missing_cols
    
    # Check for missing or empty transcriptions
    missing_trans = df['transcription'].isnull().sum()
    empty_trans = (df['transcription'].str.strip() == '').sum()
    issues['missing_transcriptions'] = missing_trans
    issues['empty_transcriptions'] = empty_trans
    
    # Check label consistency
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        issues['label_distribution'] = label_counts.to_dict()
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['file_number']).sum()
    issues['duplicate_file_numbers'] = duplicates
    
    return issues

def clean_data(df):
    """
    Clean the DataFrame by removing problematic rows and standardizing labels.
    
    Args:
        df: DataFrame to clean
    
    Returns:
        DataFrame: Cleaned DataFrame
    """
    print("Cleaning data...")
    
    # Store original size
    original_size = len(df)
    
    # Drop rows with missing transcriptions
    df = df.dropna(subset=['transcription'])
    
    # Drop rows with empty transcriptions
    df = df[df['transcription'].str.strip() != '']
    
    # Standardize labels (convert 'bona-fide'/'spoof' to 'real'/'fake')
    if 'label' in df.columns:
        df['label'] = df['label'].replace({
            'bona-fide': 'real', 
            'spoof': 'fake',
            'BONA-FIDE': 'real',
            'SPOOF': 'fake'
        })
    
    # Drop duplicates based on file_number
    df = df.drop_duplicates(subset=['file_number'])
    
    # Reset index
    df = df.reset_index(drop=True)
    
    final_size = len(df)
    removed_rows = original_size - final_size
    
    print(f"Removed {removed_rows} problematic rows")
    print(f"Final dataset size: {final_size} rows")
    
    return df

def main():
    """Main function to prepare data."""
    parser = argparse.ArgumentParser(description="Prepare transcript data for deepfake detection")
    parser.add_argument('--input', type=str, default='all_transcripts_large.csv', 
                       help='Input CSV file path')
    parser.add_argument('--output', type=str, default='all_transcripts_large_clean.csv',
                       help='Output CSV file path')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check data quality without cleaning')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
        return
    
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    print(f"Original dataset size: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check data quality
    print("\n=== Data Quality Check ===")
    issues = check_data_quality(df)
    
    for issue_type, value in issues.items():
        if issue_type == 'missing_columns':
            print(f"Missing required columns: {value}")
        elif issue_type == 'missing_transcriptions':
            print(f"Missing transcriptions: {value}")
        elif issue_type == 'empty_transcriptions':
            print(f"Empty transcriptions: {value}")
        elif issue_type == 'label_distribution':
            print(f"Label distribution: {value}")
        elif issue_type == 'duplicate_file_numbers':
            print(f"Duplicate file numbers: {value}")
    
    # Clean data if not check-only
    if not args.check_only:
        print("\n=== Cleaning Data ===")
        df_clean = clean_data(df)
        
        # Save cleaned data
        print(f"\nSaving cleaned data to {args.output}...")
        df_clean.to_csv(args.output, index=False)
        
        # Final quality check
        print("\n=== Final Quality Check ===")
        final_issues = check_data_quality(df_clean)
        
        for issue_type, value in final_issues.items():
            if issue_type == 'label_distribution':
                print(f"Final label distribution: {value}")
            elif issue_type == 'duplicate_file_numbers':
                if value == 0:
                    print("No duplicate file numbers")
                else:
                    print(f"Still have {value} duplicate file numbers")
        
        print(f"\nData preparation complete!")
        print(f"Use '{args.output}' for feature extraction and modeling.")

if __name__ == "__main__":
    main() 