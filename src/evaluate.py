# src/evaluate.py
"""
Complete evaluation script for Credit Card Fraud Detection GRU Model
Works with MLflow tracking and DVC integration
"""

import os
import sys
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Metrics and visualization
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, auc, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# MLflow for experiment tracking
import mlflow
import mlflow.pytorch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

class CreditCardGRU(nn.Module):
    """
    GRU model for credit card fraud detection - same as training
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(CreditCardGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Input projection layer
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 4)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Project input features
        x = self.input_projection(x)
        x = self.relu(x)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # GRU forward pass
        gru_out, _ = self.gru(x, h0)
        
        # Use the last output of the sequence
        last_output = gru_out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(last_output)
        
        # Fully connected layers with batch normalization
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Output layer
        out = self.fc3(out)
        
        return out

def load_test_data(data_path="Data/processed"):
    """
    Load test data from processed files
    """
    logger.info(f"Loading test data from {data_path}")
    
    try:
        # Try to find test data file
        files = os.listdir(data_path)
        test_files = [f for f in files if 'test' in f.lower()]
        
        if not test_files:
            # If no test file, use the third processed file
            processed_files = [f for f in files if f.startswith('creditcard_proces')]
            if len(processed_files) >= 3:
                test_file = processed_files[2]  # Assume third file is test
            else:
                raise FileNotFoundError("No test data found")
        else:
            test_file = test_files[0]
        
        logger.info(f"Loading test file: {test_file}")
        test_df = pd.read_csv(os.path.join(data_path, test_file))
        
        # Separate features and target
        if 'Class' in test_df.columns:
            X_test = test_df.drop('Class', axis=1).values
            y_test = test_df['Class'].values
        else:
            # Assume last column is target
            X_test = test_df.iloc[:, :-1].values
            y_test = test_df.iloc[:, -1].values
        
        logger.info(f"Test data loaded: {X_test.shape}, Labels: {y_test.shape}")
        logger.info(f"Class distribution: Normal={np.sum(y_test==0)}, Fraud={np.sum(y_test==1)}")
        
        return X_test, y_test
        
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return None, None

def load_trained_model(model_path, model_info_path, device):
    """
    Load trained model from saved files
    """
    logger.info(f"Loading model from {model_path}")
    
    try:
        # Load model metadata
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        
        # Extract model parameters
        if 'best_params' in model_info:
            params = model_info['best_params']
        else:
            # Default parameters if not found
            params = {
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.3
            }
        
        # Determine input size from model info or data
        if 'model_architecture' in model_info:
            input_size = model_info['model_architecture'].get('input_size', 30)
        else:
            input_size = 30  # Default for credit card data
        
        # Create model
        model = CreditCardGRU(
            input_size=input_size,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            num_classes=2,
            dropout=params['dropout']
        )
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully!")
        logger.info(f"Model parameters: {params}")
        
        return model, model_info
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None

def predict_with_model(model, X_test, device, batch_size=64):
    """
    Generate predictions with the trained model
    """
    logger.info("Generating predictions...")
    
    # Create test dataset and loader
    test_dataset = TensorDataset(torch.FloatTensor(X_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    predictions = []
    probabilities = []
    
    model.eval()
    with torch.no_grad():
        for (data,) in test_loader:
            data = data.to(device)
            outputs = model(data)
            
            # Get probabilities and predictions
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    return np.array(predictions), np.array(probabilities)

def calculate_metrics(y_true, y_pred, y_proba):
    """
    Calculate comprehensive evaluation metrics
    """
    logger.info("Calculating evaluation metrics...")
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # ROC AUC and PR AUC
    try:
        roc_auc = roc_auc_score(y_true, y_proba[:, 1])
        pr_auc = average_precision_score(y_true, y_proba[:, 1])
    except:
        roc_auc = 0.0
        pr_auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'normal_precision': float(precision_per_class[0]),
        'normal_recall': float(recall_per_class[0]),
        'normal_f1': float(f1_per_class[0]),
        'fraud_precision': float(precision_per_class[1]),
        'fraud_recall': float(recall_per_class[1]),
        'fraud_f1': float(f1_per_class[1]),
        'confusion_matrix': cm.tolist(),
        'true_negatives': int(cm[0, 0]),
        'false_positives': int(cm[0, 1]),
        'false_negatives': int(cm[1, 0]),
        'true_positives': int(cm[1, 1])
    }
    
    return metrics

def create_evaluation_plots(y_true, y_pred, y_proba, save_dir='reports'):
    """
    Create comprehensive evaluation plots
    """
    logger.info("Creating evaluation plots...")
    
    # Create reports directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.rcParams['figure.figsize'] = (15, 12)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'],
                ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Predicted Label')
    axes[0, 0].set_ylabel('True Label')
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba[:, 1])
    pr_auc = auc(recall_curve, precision_curve)
    
    axes[0, 2].plot(recall_curve, precision_curve, color='blue', lw=2,
                    label=f'PR curve (AUC = {pr_auc:.3f})')
    axes[0, 2].set_xlabel('Recall')
    axes[0, 2].set_ylabel('Precision')
    axes[0, 2].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[0, 2].legend(loc="lower left")
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Class Distribution
    true_counts = np.bincount(y_true)
    pred_counts = np.bincount(y_pred)
    
    x = np.arange(2)
    width = 0.35
    
    axes[1, 0].bar(x - width/2, true_counts, width, label='True', alpha=0.7, color='skyblue')
    axes[1, 0].bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.7, color='orange')
    axes[1, 0].set_xlabel('Class')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('True vs Predicted Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(['Normal', 'Fraud'])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Probability Distribution
    fraud_probs = y_proba[y_true == 1, 1]  # Fraud probabilities for actual fraud cases
    normal_probs = y_proba[y_true == 0, 1]  # Fraud probabilities for normal cases
    
    axes[1, 1].hist(normal_probs, bins=50, alpha=0.7, label='Normal', color='blue')
    axes[1, 1].hist(fraud_probs, bins=50, alpha=0.7, label='Fraud', color='red')
    axes[1, 1].set_xlabel('Fraud Probability')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Fraud Probability Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Performance Summary
    metrics_text = f"""
    Accuracy: {accuracy_score(y_true, y_pred):.4f}
    Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}
    Recall: {recall_score(y_true, y_pred, average='weighted'):.4f}
    F1-Score: {f1_score(y_true, y_pred, average='weighted'):.4f}
    ROC AUC: {roc_auc:.4f}
    PR AUC: {pr_auc:.4f}
    
    Fraud Detection:
    Precision: {precision_score(y_true, y_pred, average=None)[1]:.4f}
    Recall: {recall_score(y_true, y_pred, average=None)[1]:.4f}
    F1-Score: {f1_score(y_true, y_pred, average=None)[1]:.4f}
    """
    
    axes[1, 2].text(0.1, 0.9, metrics_text, fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                    transform=axes[1, 2].transAxes)
    axes[1, 2].set_title('Performance Summary', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/evaluation_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Evaluation plots saved to {save_dir}/evaluation_plots.png")

def generate_classification_report(y_true, y_pred, save_dir='reports'):
    """
    Generate detailed classification report
    """
    os.makedirs(save_dir, exist_ok=True)
    
    report = classification_report(y_true, y_pred, 
                                target_names=['Normal', 'Fraud'], 
                                digits=4)
    
    report_path = f'{save_dir}/classification_report.txt'
    with open(report_path, 'w') as f:
        f.write("Credit Card Fraud Detection - Classification Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    
    logger.info(f"Classification report saved to {report_path}")
    return report

def evaluate_model(model_path=None, model_info_path=None, data_path="Data/processed", 
                save_reports=True, experiment_name="credit_card_fraud_detection"):
    """
    Complete model evaluation pipeline
    """
    logger.info("Starting model evaluation...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Auto-detect model files if not provided
    if model_path is None:
        model_files = []
        for root, dirs, files in os.walk('.'):
            model_files.extend([os.path.join(root, f) for f in files if f.endswith('.pth')])
        
        if model_files:
            model_path = model_files[-1]  # Use the latest model
            logger.info(f"Auto-detected model: {model_path}")
        else:
            raise FileNotFoundError("No model file found. Please train a model first.")
    
    if model_info_path is None:
        info_files = []
        for root, dirs, files in os.walk('.'):
            info_files.extend([os.path.join(root, f) for f in files 
                            if ('model_info' in f or 'metadata' in f) and f.endswith('.json')])
        
        if info_files:
            model_info_path = info_files[-1]
            logger.info(f"Auto-detected model info: {model_info_path}")
        else:
            logger.warning("No model info file found. Using default parameters.")
            # Create a temporary model info
            model_info_path = 'temp_model_info.json'
            with open(model_info_path, 'w') as f:
                json.dump({
                    'best_params': {
                        'hidden_size': 128,
                        'num_layers': 2,
                        'dropout': 0.3
                    }
                }, f)
    
    # MLflow setup
    try:
        mlflow.set_experiment(experiment_name)
        mlflow_available = True
    except:
        logger.warning("MLflow not available. Skipping MLflow logging.")
        mlflow_available = False
    
    if mlflow_available:
        with mlflow.start_run(run_name=f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            return _run_evaluation(model_path, model_info_path, data_path, device, save_reports, mlflow_available)
    else:
        return _run_evaluation(model_path, model_info_path, data_path, device, save_reports, mlflow_available)

def _run_evaluation(model_path, model_info_path, data_path, device, save_reports, mlflow_available):
    """
    Internal evaluation function
    """
    # Load test data
    X_test, y_test = load_test_data(data_path)
    if X_test is None:
        raise ValueError("Could not load test data")
    
    # Load trained model
    model, model_info = load_trained_model(model_path, model_info_path, device)
    if model is None:
        raise ValueError("Could not load trained model")
    
    # Log parameters to MLflow if available
    if mlflow_available:
        mlflow.log_params({
            'model_path': model_path,
            'test_samples': len(X_test),
            'evaluation_date': datetime.now().isoformat()
        })
    
    # Generate predictions
    y_pred, y_proba = predict_with_model(model, X_test, device)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    
    # Log metrics to MLflow if available
    if mlflow_available:
        mlflow.log_metrics(metrics)
    
    if save_reports:
        # Create evaluation plots
        create_evaluation_plots(y_test, y_pred, y_proba)
        
        # Generate classification report
        generate_classification_report(y_test, y_pred)
        
        # Save detailed evaluation results
        evaluation_results = {
            'model_info': model_info,
            'test_metrics': metrics,
            'evaluation_date': datetime.now().isoformat(),
            'test_samples': int(len(X_test)),
            'model_path': model_path
        }
        
        os.makedirs('reports', exist_ok=True)
        with open('reports/evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Create metrics file for DVC
        dvc_metrics = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'roc_auc': metrics['roc_auc'],
            'fraud_precision': metrics['fraud_precision'],
            'fraud_recall': metrics['fraud_recall'],
            'fraud_f1': metrics['fraud_f1']
        }
        
        with open('reports/test_metrics.json', 'w') as f:
            json.dump(dvc_metrics, f, indent=2)
        
        if mlflow_available:
            mlflow.log_artifacts('reports', 'evaluation_reports')
    
    # Print results
    print("\n" + "="*70)
    print("CREDIT CARD FRAUD DETECTION - EVALUATION RESULTS")
    print("="*70)
    print(f"Test Samples: {len(X_test):,}")
    print(f"Normal Transactions: {np.sum(y_test == 0):,}")
    print(f"Fraud Transactions: {np.sum(y_test == 1):,}")
    print(f"Fraud Ratio: {np.mean(y_test):.4f}")
    print("-" * 70)
    print("OVERALL PERFORMANCE:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    print("-" * 70)
    print("FRAUD DETECTION PERFORMANCE:")
    print(f"  Precision: {metrics['fraud_precision']:.4f} (How many predicted frauds were actually frauds)")
    print(f"  Recall:    {metrics['fraud_recall']:.4f} (How many actual frauds were detected)")
    print(f"  F1-Score:  {metrics['fraud_f1']:.4f} (Balanced measure)")
    print("-" * 70)
    print("CONFUSION MATRIX:")
    print(f"  True Negatives:  {metrics['true_negatives']:,} (Correctly identified normal)")
    print(f"  False Positives: {metrics['false_positives']:,} (Incorrectly flagged as fraud)")
    print(f"  False Negatives: {metrics['false_negatives']:,} (Missed frauds)")
    print(f"  True Positives:  {metrics['true_positives']:,} (Correctly identified frauds)")
    print("="*70)
    
    if save_reports:
        print("\nReports saved to:")
        print("  - reports/evaluation_plots.png")
        print("  - reports/classification_report.txt")
        print("  - reports/evaluation_results.json")
        print("  - reports/test_metrics.json")
    
    logger.info("Evaluation completed successfully!")
    
    return model, metrics, evaluation_results if save_reports else None

# Convenience function for quick evaluation
def quick_evaluate(model_path=None, data_path="Data/processed"):
    """
    Quick evaluation without MLflow logging
    """
    return evaluate_model(model_path=model_path, data_path=data_path, 
                        save_reports=True, experiment_name="quick_evaluation")

if __name__ == "__main__":
    # Run evaluation if script is called directly
    try:
        model, metrics, results = evaluate_model()
        print(f"\nEvaluation completed! Main metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Fraud Detection F1: {metrics['fraud_f1']:.4f}")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. A trained model (.pth file)")
        print("2. Test data in Data/processed/")
        print("3. Model info/metadata file (.json)")