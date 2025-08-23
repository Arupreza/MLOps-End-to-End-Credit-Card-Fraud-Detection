# Credit Card Fraud Detection with MLOps Pipeline

A complete MLOps pipeline for credit card fraud detection using GRU (Gated Recurrent Unit) neural networks, with automated hyperparameter optimization using Optuna, experiment tracking with MLflow, and data versioning with DVC.

## 🏗️ Architecture Overview

```
Data → Preprocessing → Hyperparameter Optimization → Model Training → Evaluation → Deployment
  ↓         ↓                      ↓                        ↓             ↓           ↓
 DVC    DVC + MLflow         Optuna + MLflow          MLflow + DVC    DVC + MLflow  MLflow
```

## 🚀 Features

- **Deep Learning Model**: GRU-based neural network for sequence-based fraud detection
- **Automated Hyperparameter Optimization**: Optuna integration for efficient hyperparameter tuning
- **Experiment Tracking**: MLflow for comprehensive experiment management
- **Data Versioning**: DVC for data and model versioning
- **Reproducible Pipelines**: Automated pipelines with parameter tracking
- **Comprehensive Evaluation**: Detailed model evaluation with multiple metrics and visualizations

## 📋 Requirements

```bash
pip install -r requirements.txt
```

## 🗂️ Project Structure

```
credit-card-fraud-detection/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── gru_model.py              # GRU model architecture
│   ├── prepare_data.py               # Data preprocessing
dvc repro prepare_data

# Hyperparameter optimization
dvc repro hyperparameter_optimization

# Train final model
dvc repro train_final_model

# Evaluate model
dvc repro evaluate_model
```

### Monitor Experiments

```bash
# Start MLflow UI
mlflow ui

# View in browser: http://localhost:5000
```

## 📊 Pipeline Stages

### 1. Data Preparation (`prepare_data`)
- Loads raw credit card transaction data
- Performs data preprocessing and feature engineering
- Creates sequence data for GRU input
- Splits data into train/validation/test sets
- Saves preprocessed data and scaler

**Outputs:**
- `data/processed/X_train.npy`, `X_val.npy`, `X_test.npy`
- `data/processed/y_train.npy`, `y_val.npy`, `y_test.npy`
- `data/processed/scaler.pkl`
- `data/processed/data_info.json`

### 2. Hyperparameter Optimization (`hyperparameter_optimization`)
- Uses Optuna for automated hyperparameter tuning
- Optimizes GRU architecture parameters
- Tracks all trials with MLflow
- Saves best parameters and study results

**Key Hyperparameters:**
- `hidden_size`: GRU hidden layer size (32-256)
- `num_layers`: Number of GRU layers (1-4)
- `dropout`: Dropout rate (0.1-0.7)
- `learning_rate`: Learning rate (1e-5 to 1e-2)
- `batch_size`: Batch size [16, 32, 64, 128]
- `optimizer`: Optimizer type [Adam, SGD, RMSprop]

**Outputs:**
- `models/optuna_study/best_params.json`
- `models/optuna_study/study.pkl`
- `models/optuna_study/optimization_plots.png`
- `models/optuna_study/optimization_metrics.json`

### 3. Final Model Training (`train_final_model`)
- Trains final model with optimized hyperparameters
- Implements early stopping for optimal training
- Comprehensive logging with MLflow
- Saves trained model and training history

**Features:**
- Early stopping with patience
- Training/validation loss tracking
- Model checkpointing
- Comprehensive metrics logging

**Outputs:**
- `models/final_model/best_model.pth`
- `models/final_model/model_info.json`
- `models/final_model/training_metrics.json`
- `models/final_model/training_plots.png`

### 4. Model Evaluation (`evaluate_model`)
- Comprehensive model evaluation on test set
- Multiple evaluation metrics
- Visualization generation
- Detailed reporting

**Metrics:**
- Accuracy, Precision, Recall, F1-Score
- ROC AUC, Precision-Recall AUC
- Per-class performance metrics
- Confusion matrix analysis

**Outputs:**
- `reports/evaluation_report.json`
- `reports/test_metrics.json`
- `reports/confusion_matrix.png`
- `reports/roc_curve.png`
- `reports/precision_recall_curve.png`
- `reports/classification_report.txt`

## 🔬 Experiment Management

### MLflow Integration

The pipeline automatically tracks:
- **Parameters**: All hyperparameters and configuration
- **Metrics**: Training and validation metrics over time
- **Artifacts**: Models, plots, and reports
- **Models**: Versioned model registry

### DVC Integration

DVC manages:
- **Data Versioning**: Raw and processed datasets
- **Model Versioning**: Trained models and artifacts
- **Pipeline Reproducibility**: Automated dependency tracking
- **Remote Storage**: Centralized artifact storage

## 📈 Configuration

### `params.yaml` Structure

```yaml
prepare:
  train_size: 0.7
  val_size: 0.15
  test_size: 0.15
  sequence_length: 10

model:
  input_size: 12
  num_classes: 2

optuna:
  n_trials: 100
  timeout: 3600
  hyperparameters:
    hidden_size:
      type: "int"
      low: 32
      high: 256
      step: 32
    # ... other hyperparameters

training:
  final_epochs: 100
  early_stopping_patience: 15

mlflow:
  experiment_name: "credit_card_fraud_detection"
  tracking_uri: "your-mlflow-uri"

evaluation:
  metrics: ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
  plot_confusion_matrix: true
```

## 🛠️ Advanced Usage

### Custom Hyperparameter Ranges

Modify `params.yaml` to adjust hyperparameter search spaces:

```yaml
optuna:
  hyperparameters:
    hidden_size:
      type: "int"
      low: 64      # Increase minimum
      high: 512    # Increase maximum
      step: 64
```

### Extending the Pipeline

Add new stages to `dvc.yaml`:

```yaml
stages:
  model_deployment:
    cmd: python src/deploy_model.py
    deps:
    - src/deploy_model.py
    - models/final_model/best_model.pth
    outs:
    - deployment/model_api/
```

### Experiment Comparison

```bash
# Compare different experiments
dvc metrics diff

# Show parameter differences
dvc params diff

# Compare specific experiments
dvc metrics diff HEAD~1
```

## 🔍 Monitoring and Debugging

### Pipeline Status

```bash
# Check what needs to be run
dvc status

# Visualize pipeline DAG
dvc dag

# Show pipeline metrics
dvc metrics show --all-branches
```

### MLflow Tracking

```bash
# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# Query experiments programmatically
mlflow experiments list
mlflow runs list --experiment-id 1
```

### Common Issues and Solutions

1. **CUDA Out of Memory**
   - Reduce `batch_size` in `params.yaml`
   - Decrease `hidden_size` or `num_layers`

2. **Slow Optimization**
   - Reduce `n_trials` for faster testing
   - Use `timeout` parameter to limit time

3. **MLflow Connection Issues**
   - Check `MLFLOW_TRACKING_URI` environment variable
   - Verify network connectivity to MLflow server

## 📊 Results and Metrics

The pipeline generates comprehensive evaluation metrics:

- **Model Performance**: Accuracy, Precision, Recall, F1-Score
- **Business Metrics**: False Positive Rate, False Negative Rate
- **ROC Analysis**: ROC curves and AUC scores
- **Class Balance**: Per-class performance analysis

### Expected Performance

For credit card fraud detection:
- **Accuracy**: >99% (due to class imbalance)
- **Precision (Fraud)**: >80%
- **Recall (Fraud)**: >70%
- **F1-Score (Fraud)**: >75%

## 🚀 Deployment

### Model Export

```python
# Load trained model
import torch
from src.models.gru_model import CreditCardGRU

model = CreditCardGRU(...)
model.load_state_dict(torch.load('models/final_model/best_model.pth'))

# Export to ONNX
torch.onnx.export(model, dummy_input, 'model.onnx')
```

### API Deployment

Create a FastAPI service:

```python
from fastapi import FastAPI
import torch
import numpy as np

app = FastAPI()

@app.post("/predict")
async def predict(data: dict):
    # Load model and make prediction
    prediction = model(torch.tensor(data['features']))
    return {"fraud_probability": float(prediction)}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Optuna**: For efficient hyperparameter optimization
- **MLflow**: For comprehensive experiment tracking
- **DVC**: For data and model versioning
- **PyTorch**: For deep learning framework

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the documentation
- Contact the maintainers

---

**Happy ML Engineering! 🚀** preprocessing
│   ├── hyperparameter_optimization.py # Optuna optimization
│   ├── train_final_model.py          # Final model training
│   └── evaluate_model.py             # Model evaluation
├── data/
│   ├── raw/
│   │   └── creditcard.csv            # Raw dataset
│   └── processed/                    # Processed data (DVC tracked)
├── models/
│   ├── optuna_study/                 # Optuna optimization results
│   └── final_model/                  # Final trained model
├── reports/                          # Evaluation reports and plots
├── params.yaml                       # Pipeline parameters
├── dvc.yaml                         # DVC pipeline configuration
├── requirements.txt                 # Python dependencies
└── README.md
```

## 🔧 Setup

### 1. Initialize the Project

```bash
# Clone or create the project
git init
pip install -r requirements.txt

# Initialize DVC
dvc init
```

### 2. Configure DVC Remote Storage

```bash
# For AWS S3
dvc remote add -d myremote s3://your-bucket/fraud-detection

# For Google Cloud Storage
dvc remote add -d myremote gs://your-bucket/fraud-detection

# For local storage (testing)
mkdir -p ../dvc-remote
dvc remote add -d local ../dvc-remote
```

### 3. Add Raw Data

Place your `creditcard.csv` file in `data/raw/` and track it with DVC:

```bash
dvc add data/raw/creditcard.csv
git add data/raw/creditcard.csv.dvc .gitignore
git commit -m "Add raw credit card dataset"
```

### 4. Configure MLflow

Update the MLflow configuration in your environment or `params.yaml`:

```yaml
mlflow:
  experiment_name: "credit_card_fraud_detection"
  tracking_uri: "https://your-mlflow-server.com"  # or "file:./mlruns"
```

## 🚀 Usage

### Run the Complete Pipeline

```bash
# Execute the full MLOps pipeline
dvc repro

# Check pipeline status
dvc status

# View metrics
dvc metrics show
```

### Run Individual Stages

```bash
# Data