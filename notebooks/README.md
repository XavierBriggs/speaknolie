# Notebooks

## `deepfake_analysis.ipynb`

Interactive notebook for exploring the deepfake detection results.

**What it does:**
- Loads model metrics from `visualizations/` CSV files
- Displays the model comparison and confusion matrix plots inline
- Shows detailed per-model metrics (accuracy, precision, recall, F1)

**Usage:**
```bash
cd notebooks
jupyter notebook deepfake_analysis.ipynb
```

**Dependencies:** `pandas`, `IPython` (included with Jupyter)

No additional data files are needed beyond the CSV and PNG files already in `visualizations/`.
