import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, lars_path
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

print("--- Testing Linear Regression & LARS ---")

# 1. Test Linear Regression
print("\n1. Testing Linear Regression...")
X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print(f"Linear Regression MSE: {mse:.4f}")
if mse < 1.0:
    print("Linear Regression Test: PASSED")
else:
    print("Linear Regression Test: FAILED (MSE too high)")

# 2. Test LARS Path
print("\n2. Testing LARS Path...")
try:
    alphas, active, coefs = lars_path(X, y, method='lar')
    print(f"LARS Path generated successfully.")
    print(f"Alphas shape: {alphas.shape}")
    print(f"Coefs shape: {coefs.shape}")
    print("LARS Path Test: PASSED")
except Exception as e:
    print(f"LARS Path Test: FAILED with error: {e}")
