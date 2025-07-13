"""
save_results.py
Helper script to save training results with proper naming for comparison.
"""

import shutil
import os

def save_results_with_name(model_type):
    """Save training results with model type in filename."""
    if os.path.exists('training_results.pkl'):
        new_name = f'training_results_{model_type}.pkl'
        shutil.copy('training_results.pkl', new_name)
        print(f"Saved results as {new_name}")
    else:
        print("No training_results.pkl found!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
        save_results_with_name(model_type)
    else:
        print("Usage: python save_results.py <model_type>")
        print("Example: python save_results.py tfidf") 