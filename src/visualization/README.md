# Visualization (`src/visualization/`)

This directory contains tools for creating visualizations and analyzing results from the Audio Deepfake Detection project.

## Files

### `plot_results.py`
**Purpose**: Core visualization functions for model results

**Functions**:
- `plot_confusion_matrix()`: Detailed confusion matrix with metrics
- `plot_learning_curves()`: Training/validation loss and accuracy curves
- `plot_model_comparison()`: Compare multiple models side-by-side
- `plot_feature_importance()`: Feature importance analysis
- `plot_error_analysis()`: Error type distribution and examples

**Usage**:
```bash
# Create confusion matrix
python src/visualization/plot_results.py --results training_results_bert.pkl

# Compare multiple models
python src/visualization/plot_results.py --compare_models
```

### `create_all_plots.py`
**Purpose**: Comprehensive plotting script that generates all visualizations

**Features**:
- Automatic loading of all model results
- Creates comparison plots
- Saves all visualizations to `visualizations/` directory
- Generates summary reports

**Usage**:
```bash
python src/visualization/create_all_plots.py
```

**Output**:
- `visualizations/model_comparison.png`
- `visualizations/tfidf_confusion_matrix.png`
- `visualizations/bert_confusion_matrix.png`
- `visualizations/model_summary.csv`

### `generate_report.py`
**Purpose**: Generate comprehensive project report

**Features**:
- Creates professional markdown report
- Includes all performance metrics
- Provides research insights and conclusions
- Saves as `project_report.md`

**Usage**:
```bash
python src/visualization/generate_report.py
```

## Visualization Types

### 1. Confusion Matrix
**Purpose**: Show prediction accuracy for each class

**Features**:
- Color-coded heatmap
- Annotated with actual counts
- Accuracy metric overlay
- Clear true vs predicted labels

**Example**:
```
                Predicted
              Real    Fake
Actual Real   3633    348
      Fake    993    1366
```

### 2. Model Comparison
**Purpose**: Compare performance across different models

**Metrics**:
- Accuracy comparison
- Precision comparison
- Recall comparison
- F1 Score comparison

**Layout**: 2x2 subplot grid with bar charts

### 3. Learning Curves
**Purpose**: Monitor training progress

**Plots**:
- Training vs validation loss
- Training vs validation accuracy
- Early stopping indicators
- Overfitting detection

### 4. Error Analysis
**Purpose**: Understand model failure modes

**Analysis**:
- False positive vs false negative rates
- Error type distribution
- Example misclassified texts
- Error rate statistics

## Output Directory Structure

```
visualizations/
├── model_comparison.png          # Side-by-side model comparison
├── tfidf_confusion_matrix.png    # TF-IDF confusion matrix
├── bert_confusion_matrix.png     # BERT confusion matrix
├── tfidf_metrics.csv            # TF-IDF performance metrics
├── bert_metrics.csv             # BERT performance metrics
└── model_summary.csv            # Combined metrics summary
```

## Customization Options

### Plot Styling
```python
# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Custom colors
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# Figure sizes
figsize = (15, 10)  # Large plots
figsize = (8, 6)    # Standard plots
```

### Save Options
```python
# High quality
plt.savefig('plot.png', dpi=300, bbox_inches='tight')

# Multiple formats
plt.savefig('plot.pdf')
plt.savefig('plot.svg')
```

## Performance Metrics Visualization

### Accuracy Metrics
- **Overall Accuracy**: Percentage of correct predictions
- **Per-Class Accuracy**: Accuracy for real vs fake
- **Balanced Accuracy**: Average of per-class accuracies

### Precision-Recall Analysis
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

### Error Analysis
- **False Positive Rate**: Real → Fake misclassifications
- **False Negative Rate**: Fake → Real misclassifications
- **Error Distribution**: Visual breakdown of error types

## Interactive Analysis

### Jupyter Notebook
The `notebooks/deepfake_analysis.ipynb` provides interactive analysis:
- Load and explore results
- Create custom visualizations
- Perform detailed error analysis
- Generate insights and conclusions

### Usage
```python
# Load results
with open('training_results_bert.pkl', 'rb') as f:
    results = pickle.load(f)

# Create custom plots
plot_confusion_matrix(results['test_metrics']['confusion_matrix'], 
                     ['Real', 'Fake'], 
                     title="BERT Model Results")
```

## Best Practices

### For Publication
1. **High Resolution**: Use 300 DPI for print quality
2. **Color Blind Friendly**: Use accessible color palettes
3. **Clear Labels**: Descriptive titles and axis labels
4. **Consistent Style**: Use same style across all plots

### For Analysis
1. **Multiple Views**: Create different plot types
2. **Error Analysis**: Always include confusion matrices
3. **Comparison**: Show multiple models when available
4. **Context**: Include baseline comparisons

### For Presentation
1. **Simple Plots**: Focus on key insights
2. **Clear Titles**: Descriptive plot titles
3. **Annotated**: Add key metrics as text
4. **Consistent**: Use project color scheme

## Troubleshooting

### Common Issues
1. **Missing Results**: Check file paths and names
2. **Memory Errors**: Reduce figure sizes
3. **Style Issues**: Import matplotlib and seaborn
4. **Save Errors**: Check directory permissions

### Debugging
```python
# Check available results
import os
print(os.listdir('.'))

# Test plotting
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [1, 4, 2])
plt.show()
```

## Integration with Other Tools

### Export Options
- **PNG**: High quality for presentations
- **PDF**: Vector format for publications
- **SVG**: Scalable for web
- **CSV**: Data for further analysis

### Report Generation
- **Markdown**: Professional project reports
- **HTML**: Web-friendly documentation
- **LaTeX**: Academic publications
- **PowerPoint**: Presentation slides 