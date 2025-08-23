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


def objectative(trial):
    # Define the hyperparameter search space

    # Model Initialization

    # Params Initialization

    # Optimizer Selection

    # Training Loop

    # Validation Loop

    # Evaluation Metrics Calculation
    pass


class CreditCardGRU(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=2, num_classes=2, dropout=0.3):
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
        self.sigmoid = nn.Sigmoid()
        
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