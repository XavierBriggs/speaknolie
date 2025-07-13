# Jupyter Notebooks (`notebooks/`)

This directory contains Jupyter notebooks for interactive analysis and exploration of the Audio Deepfake Detection project.

## Files

### `deepfake_analysis.ipynb`
**Purpose**: Comprehensive interactive analysis of deepfake detection results

**Features**:
- Load and explore model results
- Create custom visualizations
- Perform detailed error analysis
- Generate insights and conclusions
- Interactive data exploration

**Sections**:
1. **Data Loading**: Import results and setup environment
2. **Model Comparison**: Side-by-side performance analysis
3. **Confusion Matrix Analysis**: Detailed error breakdown
4. **Error Analysis**: Understanding failure modes
5. **Research Insights**: Key findings and conclusions
6. **Next Steps**: Recommendations for future work

**Usage**:
```bash
# Start Jupyter
jupyter notebook

# Open the notebook
notebooks/deepfake_analysis.ipynb
```

## Notebook Structure

### 1. Setup and Imports
```python
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Set plotting style
plt.style.use('seaborn-v0_8')
```

### 2. Data Loading
```python
# Load model results
results = {}
model_files = {
    'TF-IDF': '../training_results_tfidf.pkl',
    'BERT': '../training_results_bert.pkl'
}

for model_name, file_path in model_files.items():
    with open(file_path, 'rb') as f:
        results[model_name] = pickle.load(f)['test_metrics']
```

### 3. Model Comparison
```python
# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# ... plotting code
```

### 4. Error Analysis
```python
# Analyze confusion matrices
for model_name, metrics in results.items():
    cm = metrics['confusion_matrix']
    # ... analysis code
```

## Interactive Features

### Dynamic Visualization
- **Real-time plots**: Update visualizations as you modify parameters
- **Interactive widgets**: Sliders, dropdowns for parameter exploration
- **Custom analysis**: Create new plots and analyses on-the-fly

### Data Exploration
- **Result inspection**: Examine raw data and metrics
- **Statistical analysis**: Perform additional statistical tests
- **Feature analysis**: Explore feature importance and patterns

### Error Investigation
- **Misclassified samples**: Look at specific error cases
- **Pattern analysis**: Identify common failure modes
- **Text analysis**: Examine transcript content of errors

## Best Practices

### For Analysis
1. **Start with overview**: Load and examine all results first
2. **Create hypotheses**: Formulate questions before diving deep
3. **Document insights**: Add markdown cells with observations
4. **Save outputs**: Export important plots and analyses

### For Collaboration
1. **Clear structure**: Use markdown headers for organization
2. **Reproducible**: Include all necessary imports and setup
3. **Commented code**: Explain complex analysis steps
4. **Version control**: Track notebook changes

### For Presentation
1. **Executive summary**: Start with key findings
2. **Visual focus**: Use plots to tell the story
3. **Clear conclusions**: End with actionable insights
4. **Export options**: Save as PDF or HTML for sharing

## Environment Setup

### Required Packages
```bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn
```

### Jupyter Extensions (Optional)
```bash
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
```

### Virtual Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook
```

## Notebook Workflow

### 1. Exploration Phase
- Load all available results
- Create overview visualizations
- Identify key patterns and insights

### 2. Analysis Phase
- Deep dive into specific aspects
- Perform statistical tests
- Create detailed visualizations

### 3. Synthesis Phase
- Combine findings from different analyses
- Generate insights and conclusions
- Plan next steps

### 4. Documentation Phase
- Add explanatory markdown
- Clean up code and comments
- Export final results

## Custom Analysis Examples

### Model Performance Comparison
```python
# Compare multiple models
metrics = ['accuracy', 'precision', 'recall', 'f1']
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for i, metric in enumerate(metrics):
    values = [results[model][metric] for model in results.keys()]
    # ... plotting code
```

### Error Pattern Analysis
```python
# Analyze error types
for model_name, metrics in results.items():
    cm = metrics['confusion_matrix']
    fp_rate = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    fn_rate = cm[1, 0] / (cm[1, 0] + cm[1, 1])
    print(f"{model_name}: FP={fp_rate:.3f}, FN={fn_rate:.3f}")
```

### Feature Importance (if available)
```python
# Analyze feature importance
if hasattr(model, 'feature_importances_'):
    importance = model.feature_importances_
    top_features = np.argsort(importance)[-20:]
    # ... plotting code
```

## Troubleshooting

### Common Issues
1. **Import errors**: Check package installation
2. **File not found**: Verify file paths
3. **Memory issues**: Reduce dataset size for testing
4. **Plotting errors**: Check matplotlib backend

### Debugging Tips
```python
# Check available files
import os
print(os.listdir('../'))

# Test imports
import pandas as pd
print(pd.__version__)

# Test plotting
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [1, 4, 2])
plt.show()
```

## Integration with Project

### File Dependencies
- `../training_results_*.pkl`: Model evaluation results
- `../visualizations/`: Generated plots
- `../project_report.md`: Generated report

### Workflow Integration
1. **Run training**: Generate results files
2. **Load in notebook**: Explore and analyze
3. **Create visualizations**: Generate custom plots
4. **Export insights**: Save important findings

### Version Control
- Track notebook changes
- Include outputs in documentation
- Share insights with team
- Maintain reproducibility 