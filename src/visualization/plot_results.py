"""
plot_results.py
Visualization tools for analyzing deepfake detection model results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, classification_report
import argparse
import os

def plot_confusion_matrix(cm, labels, title="Confusion Matrix", save_path=None):
    """
    Plots a confusion matrix with detailed annotations.
    
    Args:
        cm: Confusion matrix
        labels: List of class labels
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Add performance metrics as text
    total = np.sum(cm)
    accuracy = np.trace(cm) / total
    plt.text(0.02, 0.98, f'Accuracy: {accuracy:.3f}', 
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    Plots learning curves for loss and accuracy.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_model_comparison(results_dict, save_path=None):
    """
    Compares performance metrics across different models/features.
    
    Args:
        results_dict: Dictionary with model names as keys and metrics as values
        save_path: Path to save the plot
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [results_dict[model][metric] for model in results_dict.keys()]
        models = list(results_dict.keys())
        
        bars = axes[i].bar(models, values, color=['#2E86AB', '#A23B72'])
        axes[i].set_title(f'{metric.capitalize()} Comparison', fontsize=14, fontweight='bold')
        axes[i].set_ylabel(metric.capitalize(), fontsize=12)
        axes[i].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """
    Plots feature importance for the model (works with linear models).
    
    Args:
        model: Trained model with feature_importances_ or coef_ attribute
        feature_names: List of feature names
        top_n: Number of top features to show
        save_path: Path to save the plot
    """
    # Get feature importance (works for linear models)
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])  # For binary classification
    else:
        print("Model doesn't have feature importance attributes")
        return
    
    # Get top features
    top_indices = np.argsort(importance)[-top_n:]
    top_features = [feature_names[i] for i in top_indices]
    top_importance = importance[top_indices]
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(top_features)), top_importance, color='skyblue')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, top_importance)):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{value:.4f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_error_analysis(y_true, y_pred, texts, save_path=None):
    """
    Analyzes and visualizes prediction errors.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        texts: Original text samples
        save_path: Path to save the plot
    """
    # Find misclassified samples
    errors = y_true != y_pred
    error_indices = np.where(errors)[0]
    
    if len(error_indices) == 0:
        print("No errors found!")
        return
    
    # Analyze error types
    false_positives = (y_true == 0) & (y_pred == 1)
    false_negatives = (y_true == 1) & (y_pred == 0)
    
    error_types = {
        'False Positives (Real → Fake)': np.sum(false_positives),
        'False Negatives (Fake → Real)': np.sum(false_negatives)
    }
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    bars = plt.bar(error_types.keys(), error_types.values(), 
                   color=['#FF6B6B', '#4ECDC4'])
    plt.title('Error Analysis: Types of Misclassifications', fontsize=16, fontweight='bold')
    plt.ylabel('Number of Errors', fontsize=12)
    
    # Add value labels
    for bar, value in zip(bars, error_types.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(value), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print some example errors
    print(f"\n=== Error Analysis ===")
    print(f"Total errors: {len(error_indices)}")
    print(f"Error rate: {len(error_indices)/len(y_true):.3f}")
    
    # Show some example misclassified texts
    print(f"\n=== Example Misclassified Texts ===")
    for i, idx in enumerate(error_indices[:5]):  # Show first 5 errors
        true_label = "Real" if y_true[idx] == 0 else "Fake"
        pred_label = "Real" if y_pred[idx] == 0 else "Fake"
        print(f"Error {i+1}: True={true_label}, Predicted={pred_label}")
        print(f"Text: {texts[idx][:100]}...")
        print()

def create_comprehensive_report(results_file, save_dir="visualizations"):
    """
    Creates a comprehensive visualization report from training results.
    
    Args:
        results_file: Path to training results pickle file
        save_dir: Directory to save visualizations
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Load results
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    metrics = results['test_metrics']
    
    # 1. Confusion Matrix
    cm = metrics['confusion_matrix']
    plot_confusion_matrix(cm, ['Real', 'Fake'], 
                         title="Deepfake Detection Confusion Matrix",
                         save_path=os.path.join(save_dir, 'confusion_matrix.png'))
    
    # 2. Metrics Summary
    print("=== Model Performance Summary ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # 3. Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(save_dir, 'performance_metrics.csv'), index=False)
    
    print(f"\n✅ Comprehensive report created in '{save_dir}' directory")

def main():
    """Main function for visualization."""
    parser = argparse.ArgumentParser(description="Create visualizations for deepfake detection results")
    parser.add_argument('--results', type=str, default='training_results.pkl',
                       help='Path to training results file')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--compare_models', action='store_true',
                       help='Compare multiple model results')
    
    args = parser.parse_args()
    
    if args.compare_models:
        # Compare TF-IDF vs BERT results
        results = {}
        for model_type in ['tfidf', 'bert']:
            results_file = f'training_results_{model_type}.pkl'
            if os.path.exists(results_file):
                with open(results_file, 'rb') as f:
                    results[model_type] = pickle.load(f)['test_metrics']
        
        if len(results) > 1:
            plot_model_comparison(results, 
                                save_path=os.path.join(args.output_dir, 'model_comparison.png'))
        else:
            print("Need at least 2 model results to compare")
    else:
        # Create comprehensive report for single model
        create_comprehensive_report(args.results, args.output_dir)

if __name__ == "__main__":
    main() 