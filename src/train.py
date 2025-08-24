import logging
import pickle
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import yaml
import sys
import mlflow
from mlflow.models import infer_signature
from urllib.parse import urlparse
import os
import optuna

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# MLflow configuration
os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/Arupreza/MlOps_End_to_End.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "Arupreza"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "0531a59c013804c71d1476a0ee381da8cd70f3e1"

# Set MLflow tracking URI
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

class CreditCardGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        """
        GRU model for credit card fraud detection
        Args:
            input_size: Number of input features (12 in your case)
            hidden_size: Number of hidden units in GRU
            num_layers: Number of GRU layers
            num_classes: Number of output classes (2 for binary classification)
            dropout: Dropout rate for regularization
        """
        super(CreditCardGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Input projection layer (optional, for feature transformation)
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
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        """
        batch_size = x.size(0)
        
        # Project input features
        x = self.input_projection(x)
        x = self.relu(x)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # GRU forward pass
        gru_out, _ = self.gru(x, h0)
        
        # Use the last output of the sequence
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_size)
        
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

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, trial_number=None):
    """Train the model and return validation metrics"""
    model.train()
    
    train_losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)
        
        # Log training progress for MLflow
        if trial_number is not None:
            mlflow.log_metric(f"train_loss_epoch_{epoch}", avg_loss, step=trial_number)
    
    # Evaluate on validation set
    model.eval()
    val_predictions = []
    val_targets = []
    val_loss = 0
    num_val_batches = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Calculate validation loss
            loss = criterion(output, target)
            val_loss += loss.item()
            num_val_batches += 1
            
            # Get predictions
            pred = output.argmax(dim=1)
            val_predictions.extend(pred.cpu().numpy())
            val_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    val_accuracy = accuracy_score(val_targets, val_predictions)
    val_precision = precision_score(val_targets, val_predictions, average='weighted', zero_division=0)
    val_recall = recall_score(val_targets, val_predictions, average='weighted', zero_division=0)
    val_f1 = f1_score(val_targets, val_predictions, average='weighted', zero_division=0)
    avg_val_loss = val_loss / num_val_batches
    
    metrics = {
        'val_accuracy': val_accuracy,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1,
        'val_loss': avg_val_loss,
        'train_losses': train_losses
    }
    
    return metrics

def objective(trial, train_loader, val_loader, input_size, num_classes, device, experiment_name):
    """Optuna objective function with MLflow integration"""
    
    # Start MLflow run for this trial
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        # Suggest hyperparameters
        hidden_size = trial.suggest_int('hidden_size', 32, 256, step=32)
        num_layers = trial.suggest_int('num_layers', 1, 4)
        dropout = trial.suggest_float('dropout', 0.1, 0.7)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        num_epochs = trial.suggest_int('num_epochs', 10, 50)
        
        # Log hyperparameters to MLflow
        mlflow.log_params({
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'trial_number': trial.number,
            'input_size': input_size,
            'num_classes': num_classes
        })
        
        # Create model
        model = CreditCardGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        ).to(device)
        
        # Log model architecture
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_params({
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        })
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
        
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        else:  # RMSprop
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        
        mlflow.log_param('optimizer', optimizer_name)
        
        # Train model
        metrics = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, trial.number)
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            'val_accuracy': metrics['val_accuracy'],
            'val_precision': metrics['val_precision'],
            'val_recall': metrics['val_recall'],
            'val_f1': metrics['val_f1'],
            'val_loss': metrics['val_loss'],
            'final_train_loss': metrics['train_losses'][-1]
        })
        
        # Log model to MLflow
        # Create a sample input for signature inference
        sample_input = next(iter(train_loader))[0][:1].cpu().numpy()
        signature = infer_signature(sample_input, np.array([[0.5, 0.5]]))  # dummy output for signature
        
        # mlflow.pytorch.log_model(
        #     pytorch_model=model,
        #     artifact_path="model",
        #     signature=signature,
        #     registered_model_name=f"{experiment_name}_trial_{trial.number}"
        # )
        
        try:
            # Just save model locally without MLflow model registry
            mlflow.log_artifacts(trial_artifacts_dir, "trial_artifacts")
            mlflow.log_text(str(model), "model_architecture.txt")
        except Exception as e:
            logger.warning(f"Could not log model artifacts: {e}")
        
        # Save trial artifacts for DVC
        trial_artifacts_dir = f"optuna_artifacts/trial_{trial.number}"
        os.makedirs(trial_artifacts_dir, exist_ok=True)
        
        # Save model state dict
        torch.save(model.state_dict(), f"{trial_artifacts_dir}/model_state_dict.pth")
        
        # Save trial parameters and metrics
        trial_data = {
            'trial_number': trial.number,
            'params': trial.params,
            'metrics': metrics,
            'mlflow_run_id': mlflow.active_run().info.run_id
        }
        
        with open(f"{trial_artifacts_dir}/trial_data.json", 'w') as f:
            json.dump(trial_data, f, indent=2, default=str)
        
        # Log artifacts to MLflow
        mlflow.log_artifacts(trial_artifacts_dir, "trial_artifacts")
        
        return metrics['val_accuracy']

def run_optuna_optimization(train_loader, val_loader, input_size, num_classes, n_trials=50, experiment_name="gru_optimization"):
    """Run Optuna hyperparameter optimization with MLflow and DVC integration"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    # Start parent MLflow run
    with mlflow.start_run(run_name=f"optuna_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log optimization parameters
        mlflow.log_params({
            'n_trials': n_trials,
            'input_size': input_size,
            'num_classes': num_classes,
            'device': str(device),
            'optimization_start_time': datetime.now().isoformat()
        })
        
        # Create study
        study_name = f"gru_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(direction='maximize', study_name=study_name)
        
        # Create callback for logging to MLflow
        def logging_callback(study, trial):
            mlflow.log_metric("best_value", study.best_value, step=trial.number)
            mlflow.log_metric("trial_value", trial.value, step=trial.number)
            
        # Optimize
        study.optimize(
            lambda trial: objective(trial, train_loader, val_loader, input_size, num_classes, device, experiment_name),
            n_trials=n_trials,
            callbacks=[logging_callback]
        )
        
        # Log best results
        mlflow.log_params(study.best_params)
        mlflow.log_metrics({
            'best_val_accuracy': study.best_value,
            'n_completed_trials': len(study.trials)
        })
        
        # Save study for DVC
        study_dir = "optuna_study"
        os.makedirs(study_dir, exist_ok=True)
        
        # Save study object
        with open(f"{study_dir}/study.pkl", 'wb') as f:
            pickle.dump(study, f)
        
        # Save study results as JSON
        study_results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'study_name': study_name,
            'trials': [
                {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name
                }
                for trial in study.trials
            ]
        }
        
        with open(f"{study_dir}/study_results.json", 'w') as f:
            json.dump(study_results, f, indent=2)
        
        # Create optimization summary plot
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Optimization history
            optuna.visualization.matplotlib.plot_optimization_history(study, ax=axes[0, 0])
            axes[0, 0].set_title('Optimization History')
            
            # Parameter importances
            optuna.visualization.matplotlib.plot_param_importances(study, ax=axes[0, 1])
            axes[0, 1].set_title('Parameter Importances')
            
            # Parallel coordinate plot
            optuna.visualization.matplotlib.plot_parallel_coordinate(study, ax=axes[1, 0])
            axes[1, 0].set_title('Parallel Coordinate Plot')
            
            # Slice plot for most important parameter
            if study.best_params:
                most_important_param = list(study.best_params.keys())[0]
                optuna.visualization.matplotlib.plot_slice(study, params=[most_important_param], ax=axes[1, 1])
                axes[1, 1].set_title(f'Slice Plot - {most_important_param}')
            
            plt.tight_layout()
            plt.savefig(f"{study_dir}/optimization_plots.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create optimization plots: {e}")
        
        # Log study artifacts
        mlflow.log_artifacts(study_dir, "optuna_study")
        
        # Print results
        print("Best trial:")
        trial = study.best_trial
        print(f"Value: {trial.value}")
        print("Params:")
        for key, value in trial.params.items():
            print(f"  {key}: {value}")
        
        return study.best_params, study

def create_best_model(best_params, input_size, num_classes):
    """Create model with best hyperparameters"""
    model = CreditCardGRU(
        input_size=input_size,
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        num_classes=num_classes,
        dropout=best_params['dropout']
    )
    return model

def save_final_model_with_dvc(model, best_params, study, experiment_name):
    """Save final model and create DVC pipeline"""
    
    # Create models directory
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Save final model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f"{models_dir}/best_gru_model_{timestamp}.pth"
    torch.save(model.state_dict(), model_path)
    
    # Save model metadata
    model_metadata = {
        'model_path': model_path,
        'best_params': best_params,
        'best_accuracy': study.best_value,
        'timestamp': timestamp,
        'experiment_name': experiment_name,
        'total_trials': len(study.trials)
    }
    
    metadata_path = f"{models_dir}/model_metadata_{timestamp}.json"
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    # Create DVC pipeline configuration
    dvc_pipeline = {
        'stages': {
            'prepare_data': {
                'cmd': 'python src/prepare_data.py',
                'deps': ['data/raw/creditcard.csv'],
                'outs': ['data/processed/train_data.pkl', 'data/processed/val_data.pkl']
            },
            'hyperparameter_optimization': {
                'cmd': f'python src/optimize_hyperparameters.py --n_trials {len(study.trials)}',
                'deps': ['src/optimize_hyperparameters.py', 'data/processed/train_data.pkl', 'data/processed/val_data.pkl'],
                'outs': ['optuna_study/', 'optuna_artifacts/'],
                'metrics': ['optuna_study/study_results.json']
            },
            'train_final_model': {
                'cmd': 'python src/train_final_model.py',
                'deps': ['src/train_final_model.py', 'optuna_study/study_results.json'],
                'outs': [model_path],
                'metrics': [metadata_path]
            }
        }
    }
    
    with open('dvc.yaml', 'w') as f:
        yaml.dump(dvc_pipeline, f, default_flow_style=False)
    
    # Create .dvcignore file
    dvcignore_content = """
            # MLflow artifacts
            mlruns/
            .mlflow/

            # Optuna database
            optuna.db

            # Temporary files
            *.tmp
            *.log
            __pycache__/
            .pytest_cache/
            *.pyc
                    """
    
    with open('.dvcignore', 'w') as f:
        f.write(dvcignore_content)
    
    logger.info(f"Final model saved to: {model_path}")
    logger.info(f"Model metadata saved to: {metadata_path}")
    logger.info("DVC pipeline created: dvc.yaml")
    
    return model_path, metadata_path

# Example usage:
if __name__ == "__main__":
    # Assuming you have your data loaders ready
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    input_size = 12  # Number of features
    num_classes = 2  # Binary classification
    experiment_name = "credit_card_gru_fraud_detection"
    
    # Run optimization with MLflow and DVC integration
    # best_params, study = run_optuna_optimization(
    #     train_loader, val_loader, input_size, num_classes, 
    #     n_trials=50, experiment_name=experiment_name
    # )
    
    # Create best model
    # best_model = create_best_model(best_params, input_size, num_classes)
    
    # Save final model with DVC integration
    # save_final_model_with_dvc(best_model, best_params, study, experiment_name)
    
    pass