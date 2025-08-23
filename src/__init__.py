# src/__init__.py
"""
Credit Card Fraud Detection with GRU, MLflow, and DVC
Author: Arupreza
"""

__version__ = "1.0.0"

# Import main modules (add specific functions as you identify them)
try:
    from . import train
    from . import evaluate
    from . import feature_selection
except ImportError as e:
    print(f"Note: Some modules not yet available: {e}")