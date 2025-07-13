"""
create_all_plots.py
Creates comprehensive visualizations for deepfake detection results.
"""

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plot_results import plot_confusion_matrix, plot_model_comparison

def create_comprehensive_visualizations():
    """Create all visualizations for the project."""
    
    # Create output directory
    output_dir = "visualizations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("🎨 Creating comprehensive visualizations...")
    
    # Load results for both models
    results = {}
    model_files = {
        'TF-IDF': 'training_results_tfidf.pkl',
        'BERT': 'training_results_bert.pkl'
    }
    
    for model_name, file_path in model_files.items():
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                results[model_name] = pickle.load(f)['test_metrics']
            print(f"✅ Loaded {model_name} results")
        else:
            print(f"⚠️  {file_path} not found")
    
    if len(results) == 0:
        print("❌ No results files found!")
        return
    
    # 1. Model Comparison (if both models available)
    if len(results) > 1:
        print("\n📊 Creating model comparison...")
        plot_model_comparison(results, 
                            save_path=os.path.join(output_dir, 'model_comparison.png'))
    
    # 2. Individual model visualizations
    for model_name, metrics in results.items():
        print(f"\n📈 Creating visualizations for {model_name}...")
        
        # Confusion matrix
        cm = metrics['confusion_matrix']
        plot_confusion_matrix(cm, ['Real', 'Fake'], 
                            title=f"{model_name} Deepfake Detection Results",
                            save_path=os.path.join(output_dir, f'{model_name.lower()}_confusion_matrix.png'))
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(output_dir, f'{model_name.lower()}_metrics.csv'), index=False)
    
    # 3. Summary report
    print(f"\n📋 Creating summary report...")
    summary_data = []
    for model_name, metrics in results.items():
        summary_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1 Score': f"{metrics['f1']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'model_summary.csv'), index=False)
    
    # Print summary
    print(f"\n🏆 Model Performance Summary:")
    print("=" * 50)
    for _, row in summary_df.iterrows():
        print(f"{row['Model']}:")
        print(f"  Accuracy: {row['Accuracy']}")
        print(f"  Precision: {row['Precision']}")
        print(f"  Recall: {row['Recall']}")
        print(f"  F1 Score: {row['F1 Score']}")
        print()
    
    print(f"✅ All visualizations saved to '{output_dir}' directory!")
    print(f"📁 Files created:")
    for file in os.listdir(output_dir):
        print(f"  - {file}")

if __name__ == "__main__":
    create_comprehensive_visualizations() 