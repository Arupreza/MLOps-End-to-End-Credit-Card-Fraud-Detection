# run_training.py - Complete Credit Card Fraud Detection Training Pipeline
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json
from datetime import datetime
import traceback

# Add src to path
sys.path.append('src')

def load_processed_data():
    """Load your processed CSV files"""
    
    print("Loading processed data files...")
    
    try:
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
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

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

def save_model_and_info(best_model, best_params, best_value, input_size, num_classes, sequence_length):
    """Save the trained model and its metadata"""
    
    print("\nSaving model and metadata...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model weights
    model_path = 'models/best_fraud_detection_model.pth'
    torch.save(best_model.state_dict(), model_path)
    
    # Create comprehensive model info
    model_info = {
        'model_path': model_path,
        'model_architecture': {
            'input_size': input_size,
            'hidden_size': best_params['hidden_size'],
            'num_layers': best_params['num_layers'],
            'num_classes': num_classes,
            'dropout': best_params['dropout']
        },
        'best_params': best_params,
        'performance': {
            'best_validation_accuracy': float(best_value),
            'accuracy_percentage': f"{best_value:.2%}"
        },
        'training_config': {
            'sequence_length': sequence_length,
            'optimizer': best_params['optimizer'],
            'learning_rate': best_params['learning_rate'],
            'batch_size': best_params['batch_size']
        },
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'framework': 'PyTorch',
            'model_type': 'GRU',
            'task': 'Credit Card Fraud Detection'
        }
    }
    
    # Save model info as JSON
    info_path = 'models/model_info.json'
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Create a simple summary file
    summary_path = 'models/model_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Credit Card Fraud Detection Model Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Best validation accuracy: {best_value:.4f} ({best_value:.2%})\n")
        f.write(f"Model architecture: GRU with {best_params['num_layers']} layers\n")
        f.write(f"Hidden size: {best_params['hidden_size']}\n")
        f.write(f"Dropout: {best_params['dropout']:.4f}\n")
        f.write(f"Optimizer: {best_params['optimizer']}\n")
        f.write(f"Learning rate: {best_params['learning_rate']:.6f}\n")
        f.write(f"Batch size: {best_params['batch_size']}\n")
        f.write("\nFiles created:\n")
        f.write(f"- {model_path}\n")
        f.write(f"- {info_path}\n")
        f.write(f"- {summary_path}\n")
    
    return model_path, info_path, summary_path

if __name__ == "__main__":
    print("=" * 60)
    print("CREDIT CARD FRAUD DETECTION - TRAINING PIPELINE")
    print("=" * 60)
    
    # Load processed data
    X_train, X_val, y_train, y_val = load_processed_data()
    
    if X_train is None:
        print("Failed to load data. Please check your data files.")
        sys.exit(1)
    
    # Create DataLoaders with sequence format
    sequence_length = 1  # For tabular data, use sequence_length=1
    train_loader, val_loader = create_dataloaders(
        X_train, X_val, y_train, y_val, 
        batch_size=32, 
        sequence_length=sequence_length
    )
    
    # Import your training functions AFTER setting up the path
    try:
        from train import *
        print("Successfully imported training functions from train.py")
    except Exception as e:
        print(f"Error importing from train.py: {e}")
        print("Please make sure your train.py contains the required functions:")
        print("- run_optuna_optimization")
        print("- create_best_model")
        sys.exit(1)
    
    # Set parameters
    input_size = X_train.shape[1]  # Number of features per time step
    num_classes = 2               # Binary classification
    n_trials = 10                # Number of optimization trials
    
    print(f"\nModel Configuration:")
    print(f"Input size (features per time step): {input_size}")
    print(f"Sequence length: {sequence_length}")
    print(f"Number of classes: {num_classes}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Optimization trials: {n_trials}")
    
    print("\nStarting Optuna hyperparameter optimization...")
    print("This may take a while depending on n_trials...")
    print("You should see trial progress messages below:")
    print("-" * 60)
    
    try:
        # Call the optimization function - handle different return types
        result = run_optuna_optimization(
            train_loader, val_loader, input_size, num_classes, n_trials
        )
        
        # Handle different return patterns from your train.py
        if isinstance(result, tuple):
            if len(result) == 2:
                best_params, study = result
                if hasattr(study, 'best_value'):
                    best_value = study.best_value
                else:
                    best_value = max([trial.value for trial in study.trials if trial.value is not None])
            elif len(result) == 3:
                best_params, study, best_value = result
            else:
                print(f"Unexpected return format from run_optuna_optimization: {len(result)} values")
                best_params = result[0]
                study = result[1] if len(result) > 1 else None
                best_value = result[2] if len(result) > 2 else 0.0
        
        elif hasattr(result, 'best_params'):
            # If result is a study object
            study = result
            best_params = study.best_params
            best_value = study.best_value
        
        else:
            # If result is just best_params (fallback)
            best_params = result
            study = None
            best_value = 0.0
            print("Warning: Could not determine best validation accuracy")
        
        print("\n" + "=" * 60)
        print("🎉 OPTIMIZATION COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 60)
        print(f"Best validation accuracy: {best_value:.4f} ({best_value:.2%})")
        print("\nOptimal hyperparameters found:")
        print("-" * 30)
        for key, value in best_params.items():
            if isinstance(value, float):
                print(f"  {key:15}: {value:.6f}")
            else:
                print(f"  {key:15}: {value}")
        
        # Create final model with best parameters
        print(f"\nCreating final model with optimal parameters...")
        try:
            best_model = create_best_model(best_params, input_size, num_classes)
            print("✅ Model created successfully!")
        except Exception as e:
            print(f"❌ Error creating model: {e}")
            print("This might be due to missing create_best_model function")
            sys.exit(1)
        
        # Save model and metadata
        try:
            model_path, info_path, summary_path = save_model_and_info(
                best_model, best_params, best_value, input_size, num_classes, sequence_length
            )
            
            print("✅ Model and metadata saved successfully!")
            print(f"\nFiles created:")
            print(f"  📁 Model weights: {model_path}")
            print(f"  📄 Model info:    {info_path}")
            print(f"  📋 Summary:       {summary_path}")
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            traceback.print_exc()
        
        # Final success message
        print("\n" + "=" * 60)
        print("🚀 TRAINING PIPELINE COMPLETED SUCCESSFULLY! 🚀")
        print("=" * 60)
        print(f"🎯 Your fraud detection model achieved: {best_value:.2%} validation accuracy")
        print("\nWhat this means:")
        if best_value > 0.98:
            print("🌟 EXCELLENT! Your model has very high accuracy")
        elif best_value > 0.95:
            print("✅ GOOD! Your model performs well")
        else:
            print("⚠️  MODERATE: Consider more trials or feature engineering")
            
        print(f"\nFor fraud detection:")
        print(f"  • {best_value:.2%} of transactions classified correctly")
        print(f"  • Model can distinguish between normal and fraudulent transactions")
        
        print(f"\nNext steps:")
        print(f"  1. Run evaluation: python src/evaluate.py")
        print(f"  2. Check detailed results in: {info_path}")
        print(f"  3. View model summary in: {summary_path}")
        
        if study and hasattr(study, 'trials'):
            print(f"\nOptimization summary:")
            print(f"  • Completed trials: {len([t for t in study.trials if t.state.name == 'COMPLETE'])}")
            print(f"  • Failed trials: {len([t for t in study.trials if t.state.name == 'FAIL'])}")
            print(f"  • Best trial: #{study.best_trial.number}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user (Ctrl+C)")
        print("Partial results may be available in MLflow if logging was enabled")
        
    except Exception as e:
        print(f"\n❌ ERROR DURING TRAINING: {e}")
        print("\nDetailed error information:")
        traceback.print_exc()
        
        print(f"\nTroubleshooting steps:")
        print(f"1. Check that your train.py contains these functions:")
        print(f"   - run_optuna_optimization")
        print(f"   - create_best_model") 
        print(f"2. Verify MLflow credentials are correct")
        print(f"3. Ensure all dependencies are installed")
        print(f"4. Check GPU memory if using CUDA")
        
        # Try to show available functions for debugging
        try:
            available_functions = [name for name in dir() if callable(getattr(sys.modules[__name__], name, None)) and not name.startswith('_')]
            training_functions = [f for f in available_functions if any(keyword in f.lower() for keyword in ['train', 'optuna', 'model', 'optimize'])]
            if training_functions:
                print(f"\nAvailable training functions found:")
                for func in training_functions:
                    print(f"  - {func}")
        except:
            pass
    
    print("=" * 60)