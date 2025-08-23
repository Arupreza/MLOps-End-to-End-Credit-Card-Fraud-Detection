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

os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/Arupreza/basic_ml_pipeline.mlflow'
os.environ['MLFLOW_TRACKING_NAME'] = 'Arupreza'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '0531a59c013804c71d1476a0ee381da8cd70f3e1'

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

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Train the model and return validation accuracy"""
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    # Evaluate on validation set
    model.eval()
    val_predictions = []
    val_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            val_predictions.extend(pred.cpu().numpy())
            val_targets.extend(target.cpu().numpy())
    
    val_accuracy = accuracy_score(val_targets, val_predictions)
    return val_accuracy

def objective(trial, train_loader, val_loader, input_size, num_classes, device):
    """Optuna objective function"""
    
    # Suggest hyperparameters
    hidden_size = trial.suggest_int('hidden_size', 32, 256, step=32)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.7)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    # Create model
    model = CreditCardGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
    
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:  # RMSprop
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    
    # Train model
    num_epochs = 10  # You can make this a hyperparameter too
    val_accuracy = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
    
    return val_accuracy

def run_optuna_optimization(train_loader, val_loader, input_size, num_classes, n_trials=50):
    """Run Optuna hyperparameter optimization"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create study
    study = optuna.create_study(direction='maximize')
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, input_size, num_classes, device),
        n_trials=n_trials
    )
    
    # Print results
    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
    
    return study.best_params

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

# Example usage:
if __name__ == "__main__":
    # Assuming you have your data loaders ready
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    input_size = 12  # Number of features
    num_classes = 2  # Binary classification
    
    # Run optimization
    # best_params = run_optuna_optimization(train_loader, val_loader, input_size, num_classes, n_trials=50)
    
    # Create best model
    # best_model = create_best_model(best_params, input_size, num_classes)
    pass