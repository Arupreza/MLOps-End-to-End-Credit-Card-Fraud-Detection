# run_training.py - YAML-Synced Credit Card Fraud Detection Training Pipeline
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json
import yaml
from datetime import datetime
import traceback

# Add src to path
sys.path.append('src')

def _deep_merge(a: dict, b: dict) -> dict:
    """Deep-merge dict b into a (a wins on type conflicts)."""
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_params():
    """Load parameters from params.yaml; supports multi-document YAML."""
    try:
        with open('params.yaml', 'r') as f:
            docs = list(yaml.safe_load_all(f))
        if not docs:
            print("❌ params.yaml is empty; using defaults.")
            return get_default_params()

        # Merge docs left→right; later docs override earlier ones
        params = {}
        for d in docs:
            if d is None:
                continue
            if not isinstance(d, dict):
                print("⚠️  Ignoring non-mapping YAML document.")
                continue
            params = _deep_merge(params, d)

        # Optional: validate essential sections/keys exist
        required_paths = [
            ("prepare", "sequence_length"),
            ("model", "input_size"),
            ("model", "num_classes"),
            ("optuna", "n_trials"),
        ]
        for path in required_paths:
            cur = params
            for key in path:
                if key not in cur:
                    raise KeyError(f"Missing key in params.yaml: {'/'.join(path)}")
                cur = cur[key]

        print("✅ Successfully loaded params.yaml (multi-doc supported)")
        return params

    except FileNotFoundError:
        print("❌ params.yaml not found! Using default parameters...")
        return get_default_params()
    except yaml.YAMLError as e:
        print(f"❌ YAML parse error: {e}\nUsing default parameters…")
        return get_default_params()
    except Exception as e:
        print(f"❌ Error loading params.yaml: {e}\nUsing default parameters…")
        return get_default_params()


def get_default_params():
    """Default parameters if YAML file is not available"""
    return {
        'prepare': {
            'sequence_length': 10,
            'random_state': 42
        },
        'model': {
            'input_size': 12,
            'num_classes': 2
        },
        'optuna': {
            'n_trials': 20,
            'timeout': 3600,
            'study_name': "gru_fraud_detection",
            'direction': "maximize"
        },
        'training': {
            'final_epochs': 50,
            'early_stopping_patience': 15
        },
        'mlflow': {
            'experiment_name': "credit_card_fraud_detection"
        }
    }

def load_processed_data(params):
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
        
        # Verify input size matches params
        actual_input_size = X_train.shape[1]
        expected_input_size = params['model']['input_size']
        
        if actual_input_size != expected_input_size:
            print(f"⚠️  Warning: Data input size ({actual_input_size}) != params.yaml input_size ({expected_input_size})")
            print(f"   Using actual data size: {actual_input_size}")
            params['model']['input_size'] = actual_input_size
        
        print(f"Original features shape: {X_train.shape}")
        print(f"Train class distribution: Normal={np.sum(y_train==0)}, Fraud={np.sum(y_train==1)}")
        print(f"Val class distribution: Normal={np.sum(y_val==0)}, Fraud={np.sum(y_val==1)}")
        
        return X_train, X_val, y_train, y_val
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

def create_sequences_for_gru(X, sequence_length):
    """
    Convert tabular data to sequence format for GRU
    Uses sequence_length from params.yaml
    """
    if sequence_length == 1:
        # For sequence_length=1, just add dimension
        X_sequences = X.reshape(X.shape[0], 1, X.shape[1])
        print(f"Converted to sequence format (tabular): {X.shape} -> {X_sequences.shape}")
    else:
        # For sequence_length > 1, create actual sequences
        sequences = []
        for i in range(len(X) - sequence_length + 1):
            seq = X[i:i + sequence_length]
            sequences.append(seq)
        X_sequences = np.array(sequences)
        print(f"Created sequences of length {sequence_length}: {X.shape} -> {X_sequences.shape}")
        print(f"Note: {len(X) - len(X_sequences)} samples lost due to sequence creation")
        
    return X_sequences

def create_dataloaders(X_train, X_val, y_train, y_val, params):
    """Create PyTorch DataLoaders using parameters from YAML"""
    
    sequence_length = params['prepare']['sequence_length']
    
    # Convert to sequences for GRU
    X_train_seq = create_sequences_for_gru(X_train, sequence_length)
    X_val_seq = create_sequences_for_gru(X_val, sequence_length)
    
    # Adjust targets if we created sequences (for sequence_length > 1)
    if sequence_length > 1:
        y_train = y_train[sequence_length-1:]  # Take corresponding labels
        y_val = y_val[sequence_length-1:]
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_seq)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val_seq)
    y_val_tensor = torch.LongTensor(y_val)
    
    print(f"Final tensor shapes: X_train={X_train_tensor.shape}, y_train={y_train_tensor.shape}")
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Use batch_size from hyperparameter suggestions or default
    default_batch_size = 32
    if 'hyperparameters' in params['optuna'] and 'batch_size' in params['optuna']['hyperparameters']:
        # Use one of the suggested batch sizes as default
        batch_choices = params['optuna']['hyperparameters']['batch_size'].get('choices', [32])
        default_batch_size = batch_choices[0] if batch_choices else 32
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=default_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=default_batch_size, shuffle=False)
    
    print(f"Created DataLoaders with batch_size={default_batch_size}")
    
    return train_loader, val_loader

def setup_mlflow(params):
    """Setup MLflow tracking using parameters from YAML"""
    try:
        import mlflow
        
        # Set MLflow configuration from YAML
        if 'mlflow' in params:
            mlflow_config = params['mlflow']
            
            # Set tracking URI if specified
            if 'tracking_uri' in mlflow_config:
                os.environ['MLFLOW_TRACKING_URI'] = mlflow_config['tracking_uri']
                mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
            
            # Set experiment name
            experiment_name = mlflow_config.get('experiment_name', 'credit_card_fraud_detection')
            mlflow.set_experiment(experiment_name)
            
            print(f"✅ MLflow configured: {experiment_name}")
            return True
        else:
            print("⚠️  No MLflow config in params.yaml")
            return False
            
    except Exception as e:
        print(f"⚠️  MLflow setup failed: {e}")
        return False

def run_optuna_optimization_with_params(train_loader, val_loader, params):
    """Run Optuna optimization using parameters from YAML"""
    
    # Import training functions
    from train import run_optuna_optimization
    
    # Extract parameters
    input_size = params['model']['input_size']
    num_classes = params['model']['num_classes']
    n_trials = params['optuna']['n_trials']
    
    print(f"Running optimization with {n_trials} trials (from params.yaml)")
    
    # Run optimization
    result = run_optuna_optimization(
        train_loader, val_loader, input_size, num_classes, n_trials
    )
    
    return result

def save_model_and_info(best_model, best_params, best_value, params):
    """Save the trained model and its metadata using YAML config"""
    
    print("\nSaving model and metadata...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model weights
    model_path = 'models/best_fraud_detection_model.pth'
    torch.save(best_model.state_dict(), model_path)
    
    # Create comprehensive model info that includes YAML config
    model_info = {
        'model_path': model_path,
        'model_architecture': {
            'input_size': params['model']['input_size'],
            'hidden_size': best_params['hidden_size'],
            'num_layers': best_params['num_layers'],
            'num_classes': params['model']['num_classes'],
            'dropout': best_params['dropout']
        },
        'best_params': best_params,
        'performance': {
            'best_validation_accuracy': float(best_value),
            'accuracy_percentage': f"{best_value:.2%}"
        },
        'training_config': {
            'sequence_length': params['prepare']['sequence_length'],
            'optimizer': best_params['optimizer'],
            'learning_rate': best_params['learning_rate'],
            'batch_size': best_params['batch_size'],
            'n_trials_completed': params['optuna']['n_trials'],
            'final_epochs': params['training']['final_epochs'],
            'early_stopping_patience': params['training']['early_stopping_patience']
        },
        'yaml_config': {
            'params_file': 'params.yaml',
            'used_params': params
        },
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'framework': 'PyTorch',
            'model_type': 'GRU',
            'task': 'Credit Card Fraud Detection',
            'mlflow_experiment': params.get('mlflow', {}).get('experiment_name', 'N/A')
        }
    }
    
    # Save model info as JSON (compatible with your evaluate.py)
    info_path = 'models/model_info.json'
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Create a YAML-compatible summary
    summary_path = 'models/model_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Credit Card Fraud Detection Model Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration source: params.yaml\n")
        f.write(f"Best validation accuracy: {best_value:.4f} ({best_value:.2%})\n")
        f.write(f"\nConfiguration from params.yaml:\n")
        f.write(f"  Sequence length: {params['prepare']['sequence_length']}\n")
        f.write(f"  Optimization trials: {params['optuna']['n_trials']}\n")
        f.write(f"  Final training epochs: {params['training']['final_epochs']}\n")
        f.write(f"\nOptimal hyperparameters found:\n")
        f.write(f"  Hidden size: {best_params['hidden_size']}\n")
        f.write(f"  Number of layers: {best_params['num_layers']}\n")
        f.write(f"  Dropout: {best_params['dropout']:.4f}\n")
        f.write(f"  Optimizer: {best_params['optimizer']}\n")
        f.write(f"  Learning rate: {best_params['learning_rate']:.6f}\n")
        f.write(f"  Batch size: {best_params['batch_size']}\n")
        f.write(f"\nFiles created:\n")
        f.write(f"  - {model_path}\n")
        f.write(f"  - {info_path}\n")
        f.write(f"  - {summary_path}\n")
    
    return model_path, info_path, summary_path

if __name__ == "__main__":
    print("=" * 60)
    print("CREDIT CARD FRAUD DETECTION - YAML-SYNCED TRAINING PIPELINE")
    print("=" * 60)
    
    # Load configuration from params.yaml
    params = load_params()
    
    print(f"\nConfiguration loaded:")
    print(f"  📁 Input size: {params['model']['input_size']}")
    print(f"  🔄 Sequence length: {params['prepare']['sequence_length']}")
    print(f"  🎯 Optimization trials: {params['optuna']['n_trials']}")
    print(f"  ⏱️  Optimization timeout: {params['optuna'].get('timeout', 'None')} seconds")
    print(f"  🧠 Study name: {params['optuna']['study_name']}")
    print(f"  📊 MLflow experiment: {params.get('mlflow', {}).get('experiment_name', 'N/A')}")
    
    # Setup MLflow
    mlflow_available = setup_mlflow(params)
    
    # Load processed data
    X_train, X_val, y_train, y_val = load_processed_data(params)
    
    if X_train is None:
        print("Failed to load data. Please check your data files.")
        sys.exit(1)
    
    # Create DataLoaders using YAML config
    train_loader, val_loader = create_dataloaders(X_train, X_val, y_train, y_val, params)
    
    # Import training functions
    try:
        from train import create_best_model
        print("✅ Successfully imported training functions from train.py")
    except Exception as e:
        print(f"❌ Error importing from train.py: {e}")
        sys.exit(1)
    
    print(f"\nStarting Optuna hyperparameter optimization...")
    print(f"Configuration from params.yaml:")
    print(f"  • Trials: {params['optuna']['n_trials']}")
    print(f"  • Direction: {params['optuna']['direction']}")
    print(f"  • Study name: {params['optuna']['study_name']}")
    if 'timeout' in params['optuna']:
        print(f"  • Timeout: {params['optuna']['timeout']} seconds")
    print("This may take a while depending on configuration...")
    print("-" * 60)
    
    try:
        # Run optimization with YAML parameters
        result = run_optuna_optimization_with_params(train_loader, val_loader, params)
        
        # Handle different return patterns
        if isinstance(result, tuple):
            if len(result) == 2:
                best_params, study = result
                best_value = study.best_value if hasattr(study, 'best_value') else 0.0
            else:
                best_params = result[0]
                study = result[1] if len(result) > 1 else None
                best_value = result[2] if len(result) > 2 else 0.0
        elif hasattr(result, 'best_params'):
            study = result
            best_params = study.best_params
            best_value = study.best_value
        else:
            best_params = result
            study = None
            best_value = 0.0
        
        print("\n" + "=" * 60)
        print("🎉 YAML-CONFIGURED OPTIMIZATION COMPLETED! 🎉")
        print("=" * 60)
        print(f"Best validation accuracy: {best_value:.4f} ({best_value:.2%})")
        print(f"Configuration used: {params['optuna']['study_name']}")
        print(f"Trials completed: {params['optuna']['n_trials']}")
        
        print("\nOptimal hyperparameters found:")
        print("-" * 30)
        for key, value in best_params.items():
            if isinstance(value, float):
                print(f"  {key:15}: {value:.6f}")
            else:
                print(f"  {key:15}: {value}")
        
        # Validate hyperparameters against YAML constraints
        if 'hyperparameters' in params['optuna']:
            print(f"\nValidating against params.yaml constraints:")
            hp_config = params['optuna']['hyperparameters']
            
            for param, value in best_params.items():
                if param in hp_config:
                    config = hp_config[param]
                    if config['type'] == 'int' and 'low' in config and 'high' in config:
                        if config['low'] <= value <= config['high']:
                            print(f"  ✅ {param}: {value} (within {config['low']}-{config['high']})")
                        else:
                            print(f"  ⚠️  {param}: {value} (outside {config['low']}-{config['high']})")
        
        # Create final model
        print(f"\nCreating final model with YAML-validated parameters...")
        best_model = create_best_model(best_params, params['model']['input_size'], params['model']['num_classes'])
        
        # Save model with YAML configuration
        model_path, info_path, summary_path = save_model_and_info(best_model, best_params, best_value, params)
        
        print("✅ Model and YAML-synced metadata saved!")
        
        # Final success message
        print("\n" + "=" * 60)
        print("🚀 YAML-SYNCED TRAINING PIPELINE COMPLETED! 🚀")
        print("=" * 60)
        print(f"🎯 Accuracy achieved: {best_value:.2%}")
        print(f"📋 Used configuration: params.yaml")
        print(f"🔬 Study: {params['optuna']['study_name']}")
        print(f"📊 MLflow: {'✅ Enabled' if mlflow_available else '❌ Disabled'}")
        
        print(f"\nFiles created (YAML-compatible):")
        print(f"  📁 Model: {model_path}")
        print(f"  📄 Info: {info_path}")
        print(f"  📋 Summary: {summary_path}")
        
        print(f"\nNext steps:")
        print(f"  1. Run evaluation: python src/evaluate.py")
        print(f"  2. Use DVC pipeline: dvc repro")
        print(f"  3. Check MLflow UI: mlflow ui")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        traceback.print_exc()
    
    print("=" * 60)