#!/usr/bin/env python3
"""
FIXED Evaluation Script - Addresses the identified issues
"""

import os
import sys
import json
import logging
import pickle
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

# MLflow configuration (same as training)
os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/Arupreza/MlOps_End_to_End.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "Arupreza"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "0531a59c013804c71d1476a0ee381da8cd70f3e1"

# Set MLflow tracking URI
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

class CreditCardGRU(nn.Module):
    """
    FIXED GRU model - addresses batch normalization issues
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
        
        # FIXED: Use LayerNorm instead of BatchNorm for more stable evaluation
        self.ln1 = nn.LayerNorm(hidden_size // 2)
        self.ln2 = nn.LayerNorm(hidden_size // 4)
    
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
        
        # Apply dropout only during training
        if self.training:
            last_output = self.dropout(last_output)
        
        # Fully connected layers with layer normalization
        out = self.fc1(last_output)
        out = self.ln1(out)
        out = self.relu(out)
        if self.training:
            out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.ln2(out)
        out = self.relu(out)
        if self.training:
            out = self.dropout(out)
        
        # Output layer
        out = self.fc3(out)
        
        return out

def load_original_test_data(force_original=True):
    """
    Load ORIGINAL imbalanced test data, not the artificially balanced version
    """
    logger.info("Looking for ORIGINAL test data...")
    
    # First, try to find the original unbalanced dataset
    original_paths = [
        "Data/raw/creditcard.csv",
        "data/raw/creditcard.csv", 
        "Data/creditcard.csv",
        "creditcard.csv"
    ]
    
    for path in original_paths:
        if os.path.exists(path):
            logger.info(f"Found original dataset: {path}")
            try:
                # Load original data
                df = pd.read_csv(path)
                logger.info(f"Original data shape: {df.shape}")
                
                # Check class distribution
                class_dist = df['Class'].value_counts()
                logger.info(f"Original class distribution:\n{class_dist}")
                fraud_ratio = class_dist[1] / len(df)
                logger.info(f"Fraud ratio: {fraud_ratio:.4f} ({fraud_ratio:.2%})")
                
                if fraud_ratio > 0.1:  # If more than 10% fraud, it's likely balanced
                    logger.warning("This appears to be a balanced dataset, not original")
                    continue
                
                # Use last 20% as test set (more realistic)
                test_size = int(0.2 * len(df))
                test_df = df.tail(test_size).copy()
                
                logger.info(f"Created test set: {len(test_df)} samples")
                test_class_dist = test_df['Class'].value_counts()
                logger.info(f"Test class distribution:\n{test_class_dist}")
                
                # Separate features and labels
                if 'Class' in test_df.columns:
                    X_test = test_df.drop('Class', axis=1).values
                    y_test = test_df['Class'].values
                else:
                    X_test = test_df.iloc[:, :-1].values
                    y_test = test_df.iloc[:, -1].values
                
                return X_test, y_test, f"original_data_{test_size}_samples"
                
            except Exception as e:
                logger.warning(f"Error loading {path}: {e}")
                continue
    
    # Fallback: Use existing processed test data but warn about issues
    logger.warning("Could not find original dataset, using processed test data")
    logger.warning("Note: This data appears to be artificially balanced (50/50)")
    
    processed_paths = ["Data/processed", "data/processed"]
    
    for data_path in processed_paths:
        if os.path.exists(data_path):
            files = os.listdir(data_path)
            test_files = [f for f in files if 'test' in f.lower() and f.endswith('.csv')]
            
            if test_files:
                test_file = test_files[0]
                test_path = os.path.join(data_path, test_file)
                
                df = pd.read_csv(test_path)
                if 'Class' in df.columns:
                    X_test = df.drop('Class', axis=1).values
                    y_test = df['Class'].values
                else:
                    X_test = df.iloc[:, :-1].values
                    y_test = df.iloc[:, -1].values
                
                return X_test, y_test, f"processed_{test_file}"
    
    raise FileNotFoundError("Could not find any test data!")

def load_model_with_compatibility_fix(model_path, best_params, device):
    """
    Load model with compatibility fixes for batch normalization issues
    """
    logger.info(f"Loading model with compatibility fixes...")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Best params: {best_params}")
    
    try:
        # Extract parameters
        input_size = best_params.get('input_size', 12)
        hidden_size = best_params.get('hidden_size', 128)
        num_layers = best_params.get('num_layers', 2)
        dropout = best_params.get('dropout', 0.3)
        
        logger.info(f"Creating model: input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}")
        
        # Load the saved state dict first
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        logger.info(f"Loaded state dict with keys: {list(state_dict.keys())[:5]}...")
        
        # Check if the state dict uses BatchNorm or LayerNorm
        has_batch_norm = any('bn1.weight' in key or 'bn2.weight' in key for key in state_dict.keys())
        has_layer_norm = any('ln1.weight' in key or 'ln2.weight' in key for key in state_dict.keys())
        
        logger.info(f"Model uses BatchNorm: {has_batch_norm}, LayerNorm: {has_layer_norm}")
        
        if has_batch_norm:
            # Original model with BatchNorm - need to use the old architecture
            logger.info("Using original BatchNorm architecture for compatibility")
            
            class OriginalCreditCardGRU(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
                    super(OriginalCreditCardGRU, self).__init__()
                    self.input_size = input_size
                    self.hidden_size = hidden_size
                    self.num_layers = num_layers
                    self.num_classes = num_classes
                    
                    self.input_projection = nn.Linear(input_size, hidden_size)
                    self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, 
                                     num_layers=num_layers, batch_first=True,
                                     dropout=dropout if num_layers > 1 else 0, bidirectional=False)
                    self.dropout = nn.Dropout(dropout)
                    self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
                    self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
                    self.fc3 = nn.Linear(hidden_size // 4, num_classes)
                    self.relu = nn.ReLU()
                    self.bn1 = nn.BatchNorm1d(hidden_size // 2)
                    self.bn2 = nn.BatchNorm1d(hidden_size // 4)
            
                def forward(self, x):
                    batch_size = x.size(0)
                    x = self.input_projection(x)
                    x = self.relu(x)
                    h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
                    gru_out, _ = self.gru(x, h0)
                    last_output = gru_out[:, -1, :]
                    
                    # CRITICAL FIX: Set batch norm to eval mode and handle single samples
                    out = self.dropout(last_output) if self.training else last_output
                    out = self.fc1(out)
                    
                    # Fix for BatchNorm with single samples
                    if out.size(0) == 1:
                        # For single sample, skip batch norm
                        out = out
                    else:
                        out = self.bn1(out)
                    out = self.relu(out)
                    
                    out = self.dropout(out) if self.training else out
                    out = self.fc2(out)
                    
                    if out.size(0) == 1:
                        out = out
                    else:
                        out = self.bn2(out)
                    out = self.relu(out)
                    
                    out = self.dropout(out) if self.training else out
                    out = self.fc3(out)
                    return out
            
            model = OriginalCreditCardGRU(input_size, hidden_size, num_layers, 2, dropout)
        else:
            # Use the new LayerNorm architecture
            model = CreditCardGRU(input_size, hidden_size, num_layers, 2, dropout)
        
        # Load the state dict
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()  # CRITICAL: Set to evaluation mode
        
        # Test the model to see if it produces varied outputs
        logger.info("Testing model output variation...")
        test_input = torch.randn(10, 1, input_size).to(device)
        
        with torch.no_grad():
            test_outputs = model(test_input)
            
        # Check if outputs are all identical (indicating broken model)
        if torch.allclose(test_outputs[0:1].repeat(len(test_outputs), 1), test_outputs, atol=1e-6):
            logger.error("🚨 CRITICAL: Model produces identical outputs for different inputs!")
            logger.error("This suggests the model was not properly trained or has issues")
            
            # Try to diagnose the issue
            logger.info("Model state diagnosis:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    logger.info(f"  {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")
                    
            return None, None
        else:
            logger.info("✅ Model produces varied outputs - looks good!")
            output_std = test_outputs.std().item()
            logger.info(f"Output standard deviation: {output_std:.6f}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'architecture_type': 'BatchNorm' if has_batch_norm else 'LayerNorm'
        }
        
        logger.info("✅ Model loaded successfully with compatibility fixes!")
        return model, model_info
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def predict_with_fixed_model(model, X_test, device, batch_size=64):
    """
    Generate predictions with fixes for batch normalization issues
    """
    logger.info("Generating predictions with model fixes...")
    
    # Ensure proper tensor format
    if len(X_test.shape) == 2:
        X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)  # Add sequence dimension
    else:
        X_test_tensor = torch.FloatTensor(X_test)
    
    logger.info(f"Input tensor shape: {X_test_tensor.shape}")
    
    # Use larger batch size to avoid BatchNorm issues with small batches
    effective_batch_size = max(batch_size, 32)  # Minimum batch size for BatchNorm
    
    test_dataset = TensorDataset(X_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=effective_batch_size, shuffle=False)
    
    predictions = []
    probabilities = []
    
    model.eval()  # CRITICAL: Ensure evaluation mode
    
    with torch.no_grad():
        for batch_idx, (data,) in enumerate(test_loader):
            data = data.to(device)
            
            # Forward pass
            outputs = model(data)
            
            # Get probabilities and predictions
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            
            # Log progress every 100 batches
            if batch_idx % 100 == 0:
                logger.info(f"Processed batch {batch_idx}: outputs range [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
    
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    
    # Check prediction sanity
    unique_preds, pred_counts = np.unique(predictions, return_counts=True)
    logger.info(f"Prediction distribution: {dict(zip(unique_preds, pred_counts))}")
    
    if len(unique_preds) == 1:
        logger.warning(f"🚨 Model only predicts class {unique_preds[0]}!")
    
    return predictions, probabilities

def main_fixed_evaluation():
    """
    Main evaluation function with all fixes applied
    """
    logger.info("🚀 STARTING FIXED EVALUATION")
    logger.info("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # 1. Load ORIGINAL test data (not artificially balanced)
        logger.info("Step 1: Loading test data...")
        X_test, y_test, data_source = load_original_test_data()
        logger.info(f"✅ Loaded test data from: {data_source}")
        logger.info(f"Test data shape: {X_test.shape}, Labels: {y_test.shape}")
        
        # Show actual class distribution
        unique_labels, counts = np.unique(y_test, return_counts=True)
        logger.info("Class distribution:")
        for label, count in zip(unique_labels, counts):
            logger.info(f"  Class {label}: {count:,} ({count/len(y_test):.1%})")
        
        # 2. Find and load the best model
        logger.info("\nStep 2: Loading model...")
        study_path = "optuna_study/study_results.json"
        if os.path.exists(study_path):
            with open(study_path, 'r') as f:
                study_results = json.load(f)
            best_params = study_results['best_params']
            logger.info(f"Best Optuna accuracy: {study_results.get('best_value', 'N/A')}")
            
            # Find model file
            optuna_models = []
            if os.path.exists("optuna_artifacts"):
                for root, dirs, files in os.walk("optuna_artifacts"):
                    for file in files:
                        if file.endswith('.pth'):
                            optuna_models.append(os.path.join(root, file))
            
            if optuna_models:
                model_path = max(optuna_models, key=os.path.getmtime)
            else:
                raise FileNotFoundError("No Optuna model files found")
        else:
            raise FileNotFoundError("No Optuna study results found")
        
        logger.info(f"Using model: {model_path}")
        
        # 3. Load model with compatibility fixes
        model, model_info = load_model_with_compatibility_fix(model_path, best_params, device)
        if model is None:
            raise ValueError("Failed to load model - please retrain!")
        
        # 4. Generate predictions with fixes
        logger.info("\nStep 3: Generating predictions...")
        y_pred, y_proba = predict_with_fixed_model(model, X_test, device)
        
        # 5. Calculate metrics
        logger.info("\nStep 4: Calculating metrics...")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        try:
            if len(np.unique(y_test)) > 1:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                roc_auc = 0.5
        except:
            roc_auc = 0.5
        
        # Fraud-specific metrics
        fraud_precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        fraud_recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        fraud_f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # 6. Print results
        print("\n" + "="*80)
        print("🎯 FIXED CREDIT CARD FRAUD DETECTION EVALUATION")
        print("="*80)
        print(f"📊 Data source: {data_source}")
        print(f"📊 Test Dataset: {len(X_test):,} samples")
        print(f"   • Normal: {np.sum(y_test == 0):,} ({np.mean(y_test == 0):.1%})")
        print(f"   • Fraud: {np.sum(y_test == 1):,} ({np.mean(y_test == 1):.1%})")
        print(f"🤖 Model: {model_info['total_params']:,} parameters ({model_info['architecture_type']})")
        print("-" * 80)
        print("📈 PERFORMANCE METRICS:")
        print(f"   • Overall Accuracy: {accuracy:.1%}")
        print(f"   • Weighted Precision: {precision:.3f}")
        print(f"   • Weighted Recall: {recall:.3f}")
        print(f"   • Weighted F1: {f1:.3f}")
        print(f"   • ROC AUC: {roc_auc:.3f}")
        print("-" * 80)
        print("🔍 FRAUD DETECTION PERFORMANCE:")
        print(f"   • Fraud Precision: {fraud_precision:.3f}")
        print(f"   • Fraud Recall: {fraud_recall:.3f}")
        print(f"   • Fraud F1: {fraud_f1:.3f}")
        print("-" * 80)
        print("📊 CONFUSION MATRIX:")
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            print(f"   • True Negatives: {tn:,}")
            print(f"   • False Positives: {fp:,}")
            print(f"   • False Negatives: {fn:,}")
            print(f"   • True Positives: {tp:,}")
        else:
            print(f"   {cm}")
        print("="*80)
        
        # Check if results are reasonable
        if accuracy > 0.85 and roc_auc > 0.7:
            print("✅ Results look reasonable!")
        elif accuracy <= 0.6:
            print("🚨 Results still look suspicious - model may need retraining")
        else:
            print("⚠️  Results are improved but could be better")
            
        return True
        
    except Exception as e:
        logger.error(f"Fixed evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main_fixed_evaluation()
    if success:
        print("\n🎉 FIXED EVALUATION COMPLETED!")
    else:
        print("\n❌ EVALUATION STILL HAS ISSUES")
        print("💡 Consider retraining the model with proper validation")