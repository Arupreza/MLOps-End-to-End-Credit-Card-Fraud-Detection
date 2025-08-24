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
        """
        super(CreditCardGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Input projection layer with proper initialization
        self.input_projection = nn.Linear(input_size, hidden_size)
        nn.init.xavier_uniform_(self.input_projection.weight)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Initialize GRU weights properly
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, num_classes)
        
        # Initialize FC weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
        # Activation functions
        self.relu = nn.ReLU()
        
        # Use LayerNorm instead of BatchNorm for better stability
        self.ln1 = nn.LayerNorm(hidden_size // 2)
        self.ln2 = nn.LayerNorm(hidden_size // 4)
    
    def forward(self, x):
        """
        Forward pass with improved stability
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
        last_output = gru_out[:, -1, :]
        
        # Fully connected layers with layer normalization
        out = self.fc1(last_output)
        out = self.ln1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.ln2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Output layer
        out = self.fc3(out)
        
        return out

def load_data_loaders():
    """
    FIXED: Load your processed data and create data loaders
    """
    logger.info("Loading processed data...")
    
    # Load your processed CSV files
    train_df = pd.read_csv("Data/processed/creditcard_processed_train.csv")
    val_df = pd.read_csv("Data/processed/creditcard_processed_val.csv")
    
    logger.info(f"Train data shape: {train_df.shape}")
    logger.info(f"Validation data shape: {val_df.shape}")
    
    # Extract features and labels
    def extract_features_labels(df):
        if 'Class' in df.columns:
            X = df.drop('Class', axis=1).values
            y = df['Class'].values
        else:
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
        return X, y
    
    X_train, y_train = extract_features_labels(train_df)
    X_val, y_val = extract_features_labels(val_df)
    
    # Convert to tensors and add sequence dimension
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)  # Add sequence dimension
    y_train_tensor = torch.LongTensor(y_train)
    
    X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1)
    y_val_tensor = torch.LongTensor(y_val)
    
    logger.info(f"Train tensor shape: {X_train_tensor.shape}")
    logger.info(f"Val tensor shape: {X_val_tensor.shape}")
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Check class distribution
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_val, counts_val = np.unique(y_val, return_counts=True)
    
    logger.info(f"Train class distribution: {dict(zip(unique_train, counts_train))}")
    logger.info(f"Val class distribution: {dict(zip(unique_val, counts_val))}")
    
    return train_loader, val_loader, X_train.shape[1]  # Return input_size

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, trial_number=None):
    """IMPROVED: Train the model with better monitoring"""
    logger.info(f"Training model for {num_epochs} epochs...")
    
    train_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        num_batches = 0
        correct_preds = 0
        total_preds = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Calculate training accuracy
            _, predicted = output.max(1)
            total_preds += target.size(0)
            correct_preds += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / num_batches
        train_accuracy = correct_preds / total_preds
        train_losses.append(avg_loss)
        
        # Validation phase
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
        
        # Calculate validation metrics
        val_accuracy = accuracy_score(val_targets, val_predictions)
        val_precision = precision_score(val_targets, val_predictions, average='weighted', zero_division=0)
        val_recall = recall_score(val_targets, val_predictions, average='weighted', zero_division=0)
        val_f1 = f1_score(val_targets, val_predictions, average='weighted', zero_division=0)
        avg_val_loss = val_loss / num_val_batches
        
        val_accuracies.append(val_accuracy)
        
        # Early stopping based on validation accuracy
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Log progress
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            logger.info(f"Epoch {epoch+1}/{num_epochs}:")
            logger.info(f"  Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            logger.info(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            logger.info(f"  Val F1: {val_f1:.4f}, Best Val Acc: {best_val_acc:.4f}")
        
        # Log to MLflow if trial_number provided
        if trial_number is not None:
            mlflow.log_metric(f"train_loss_epoch_{epoch}", avg_loss, step=trial_number)
            mlflow.log_metric(f"val_acc_epoch_{epoch}", val_accuracy, step=trial_number)
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    metrics = {
        'val_accuracy': best_val_acc,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1,
        'val_loss': avg_val_loss,
        'train_losses': train_losses,
        'epochs_trained': epoch + 1
    }
    
    return metrics

def objective(trial, train_loader, val_loader, input_size, num_classes, device, experiment_name):
    """FIXED: Optuna objective function with proper error handling"""
    
    # Start MLflow run for this trial
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        # Suggest hyperparameters
        hidden_size = trial.suggest_int('hidden_size', 64, 256, step=32)
        num_layers = trial.suggest_int('num_layers', 1, 3)  # Reduced range
        dropout = trial.suggest_float('dropout', 0.2, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        num_epochs = trial.suggest_int('num_epochs', 15, 40)  # Reasonable range
        
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
        
        # Calculate class weights for imbalanced data
        y_train_list = []
        for _, y_batch in train_loader:
            y_train_list.extend(y_batch.numpy())
        y_train_array = np.array(y_train_list)
        
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_array), y=y_train_array)
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
        
        # Define loss function with class weights and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
        
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        else:  # RMSprop
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        mlflow.log_param('optimizer', optimizer_name)
        
        # Train model
        try:
            metrics = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, trial.number)
        except Exception as e:
            logger.error(f"Training failed for trial {trial.number}: {e}")
            return 0.0  # Return poor score for failed trials
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            'val_accuracy': metrics['val_accuracy'],
            'val_precision': metrics['val_precision'],
            'val_recall': metrics['val_recall'],
            'val_f1': metrics['val_f1'],
            'val_loss': metrics['val_loss'],
            'final_train_loss': metrics['train_losses'][-1],
            'epochs_trained': metrics['epochs_trained']
        })
        
        # FIXED: Save trial artifacts (moved BEFORE usage)
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
        try:
            mlflow.log_artifacts(trial_artifacts_dir, "trial_artifacts")
            mlflow.log_text(str(model), "model_architecture.txt")
        except Exception as e:
            logger.warning(f"Could not log model artifacts: {e}")
        
        logger.info(f"Trial {trial.number} completed with validation accuracy: {metrics['val_accuracy']:.4f}")
        
        return metrics['val_accuracy']

def run_optuna_optimization(train_loader, val_loader, input_size, num_classes, n_trials=20, experiment_name="gru_optimization"):
    """FIXED: Run Optuna hyperparameter optimization"""
    
    logger.info(f"Starting Optuna optimization with {n_trials} trials...")
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
            logger.info(f"Trial {trial.number} completed. Best so far: {study.best_value:.4f}")
            
        # Optimize
        study.optimize(
            lambda trial: objective(trial, train_loader, val_loader, input_size, num_classes, device, experiment_name),
            n_trials=n_trials,
            callbacks=[logging_callback],
            catch=(Exception,)  # Continue optimization even if some trials fail
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
        print("\n" + "="*60)
        print("🎯 OPTUNA OPTIMIZATION COMPLETED!")
        print("="*60)
        print("Best trial:")
        trial = study.best_trial
        print(f"  Validation Accuracy: {trial.value:.4f}")
        print("  Best Parameters:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        print(f"  Total trials completed: {len(study.trials)}")
        print("="*60)
        
        return study.best_params, study

def save_final_model_with_dvc(model, best_params, study, experiment_name):
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{experiment_name}_best.pth"
    torch.save(model.state_dict(), model_path)

    meta = {
        "best_params": best_params,
        "best_trial_number": study.best_trial.number,
        "best_val_accuracy": study.best_value,
        "experiment_name": experiment_name,
        "saved_at": datetime.now().isoformat()
    }
    with open(f"models/{experiment_name}_best.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Optional: also log artifacts to MLflow if a run is active
    try:
        mlflow.log_artifact(model_path, artifact_path="final_model")
        mlflow.log_text(json.dumps(meta, indent=2), artifact_file="final_model/metadata.json")
    except Exception as e:
        logger.warning(f"MLflow logging of final model failed: {e}")


# FIXED: Actually run the training when script is executed
if __name__ == "__main__":
    logger.info("🚀 STARTING CREDIT CARD FRAUD DETECTION TRAINING")
    logger.info("="*70)
    
    try:
        # Load data loaders
        train_loader, val_loader, input_size = load_data_loaders()
        
        num_classes = 2  # Binary classification
        experiment_name = "credit_card_gru_fraud_detection_fixed"
        n_trials = 15  # Start with fewer trials for testing
        
        logger.info(f"Input size: {input_size}")
        logger.info(f"Number of classes: {num_classes}")
        logger.info(f"Number of trials: {n_trials}")
        
        # Run optimization with MLflow and DVC integration
        best_params, study = run_optuna_optimization(
            train_loader, val_loader, input_size, num_classes, 
            n_trials=n_trials, experiment_name=experiment_name
        )
        
        # Create and save best model
        best_model = CreditCardGRU(
            input_size=input_size,
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers'],
            num_classes=num_classes,
            dropout=best_params['dropout']
        )
        
        # Load the best model weights
        best_trial_dir = f"optuna_artifacts/trial_{study.best_trial.number}"
        if os.path.exists(f"{best_trial_dir}/model_state_dict.pth"):
            best_model.load_state_dict(torch.load(f"{best_trial_dir}/model_state_dict.pth"))
            logger.info(f"Loaded best model weights from trial {study.best_trial.number}")
        
        # Save final model with DVC integration
        save_final_model_with_dvc(best_model, best_params, study, experiment_name)
        
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("You can now run evaluation with: python src/evaluate.py")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n❌ Training failed: {e}")
        print("Check the logs above for details")