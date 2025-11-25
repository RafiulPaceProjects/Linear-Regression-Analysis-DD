import sys
import numpy as np
from sklearn.datasets import make_regression

print("--- Testing XGBoost ---")
print(f"Python Version: {sys.version}")

try:
    import xgboost as xgb
    print(f"XGBoost Version: {xgb.__version__}")
    
    # Create dummy data
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1)
    
    print("Attempting to train XGBRegressor...")
    model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X, y)
    print("Training successful!")
    
    prediction = model.predict(X[:1])
    print(f"Prediction for first sample: {prediction}")
    
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    # Check for common libomp issue in message
    if "libomp" in str(e):
        print("\nDIAGNOSIS: Missing libomp (OpenMP) dependency.")
        print("On macOS, this is often fixed with: brew install libomp")
