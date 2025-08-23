# run_training.py (Fixed version)
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Add src to path
sys.path.append('src')

def load_processed_data():
    """Load your processed CSV files"""
    
    print("Loading processed data files...")
    
    # Load your CSV files
    train_df = pd.read_csv('Data/processed/creditcard_processed_train.csv')
    val_df = pd.read_csv('Data/processed/creditcard_processed_val.csv')
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Validation data shape: {val_df.shape}")
    
    # Check if Class column exists
    if 'Class' in train_df.columns:
        X_train = train_df.drop('Class', axis=1).values
        y_train = train_df['Class'].values
        X_val = val_df.drop('Class', axis=1).values
        y_val = val_df['Class'].values
        print("Found 'Class' column as target")
    else:
        # Assume last column is target
        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values.astype(int)
        X_val = val_df.iloc[:, :-1].values  
        y_val = val_df.iloc[:, -1].values.astype(int)
        print("Using last column as target")
    
    print(f"Original features shape: {X_train.shape}")
    print(f"Train class distribution: Normal={np.sum(y_train==0)}, Fraud={np.sum(y_train==1)}")
    print(f"Val class distribution: Normal={np.sum(y_val==0)}, Fraud={np.sum(y_val==1)}")
    
    return X_train, X_val, y_train, y_val

def create_sequences_for_gru(X, sequence_length=1):
    """
    Convert tabular data to sequence format for GRU
    For tabular data, we can use sequence_length=1 (each sample is a sequence of length 1)
    """
    # Add sequence dimension: (samples, features) -> (samples, sequence_length, features)
    X_sequences = X.reshape(X.shape[0], sequence_length, X.shape[1])
    print(f"Converted to sequence format: {X.shape} -> {X_sequences.shape}")
    return X_sequences

def create_dataloaders(X_train, X_val, y_train, y_val, batch_size=32, sequence_length=1):
    """Create PyTorch DataLoaders with proper sequence format"""
    
    # Convert to sequences for GRU
    X_train_seq = create_sequences_for_gru(X_train, sequence_length)
    X_val_seq = create_sequences_for_gru(X_val, sequence_length)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_seq)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val_seq)
    y_val_tensor = torch.LongTensor(y_val)
    
    print(f"Tensor shapes: X_train={X_train_tensor.shape}, y_train={y_train_tensor.shape}")
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Created DataLoaders with batch_size={batch_size}")
    
    return train_loader, val_loader

if __name__ == "__main__":
    print("="*60)
    print("CREDIT CARD FRAUD DETECTION - TRAINING PIPELINE")
    print("="*60)
    
    # Load processed data
    X_train, X_val, y_train, y_val = load_processed_data()
    
    # Create DataLoaders with sequence format
    sequence_length = 1  # For tabular data, use sequence_length=1
    train_loader, val_loader = create_dataloaders(
        X_train, X_val, y_train, y_val, 
        batch_size=32, 
        sequence_length=sequence_length
    )
    
    # Import your training functions AFTER setting up the path
    from train import *
    
    # Set parameters
    input_size = X_train.shape[1]  # Number of features per time step
    num_classes = 2               # Binary classification
    
    print(f"Input size (features per time step): {input_size}")
    print(f"Sequence length: {sequence_length}")
    print(f"Number of classes: {num_classes}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    print("\nStarting Optuna hyperparameter optimization...")
    print("This may take a while depending on n_trials...")
    
    try:
        # Call with correct function signature (no experiment_name)
        best_params, study = run_optuna_optimization(
            train_loader, val_loader, input_size, num_classes, n_trials=10
        )
        
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Best validation accuracy: {study.best_value:.4f}")
        print("Best parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        # Create and save final model
        print("\nCreating final model with best parameters...")
        best_model = create_best_model(best_params, input_size, num_classes)
        
        # Simple model save
        import os
        import json
        from datetime import datetime
        
        os.makedirs('models', exist_ok=True)
        
        # Save model
        model_path = 'models/best_fraud_detection_model.pth'
        torch.save(best_model.state_dict(), model_path)
        
        # Save model info
        model_info = {
            'model_path': model_path,
            'best_params': best_params,
            'best_accuracy': float(study.best_value),
            'input_size': input_size,
            'num_classes': num_classes,
            'sequence_length': sequence_length,
            'timestamp': datetime.now().isoformat()
        }
        
        info_path = 'models/model_info.json'
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Model saved to: {model_path}")
        print(f"Model info saved to: {info_path}")
        print("\nTraining pipeline completed successfully!")
        print("You can now run evaluation using: python src/evaluate.py")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        
    print("="*60)