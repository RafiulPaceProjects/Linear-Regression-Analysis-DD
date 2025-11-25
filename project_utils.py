import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
except Exception:
    # Catch other errors like library loading issues (libomp)
    XGBOOST_AVAILABLE = False
import itertools

def load_data(filepath):
    """Loads the diabetes dataset."""
    try:
        df = pd.read_csv(filepath, sep="\t")
        return df
    except Exception as e:
        return None

def train_linear_model(X, y):
    """Trains a linear regression model."""
    model = LinearRegression()
    model.fit(X, y)
    return model

def train_xgboost(X, y):
    """Trains an XGBoost regressor."""
    if not XGBOOST_AVAILABLE:
        return None
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X, y)
    return model

def get_best_feature(df, target_col):
    """Finds the single best feature for predicting the target."""
    features = [c for c in df.columns if c != target_col]
    best_feature = None
    best_mse = float('inf')
    best_model = None
    
    y = df[target_col]
    
    for feature in features:
        X = df[[feature]]
        model = train_linear_model(X, y)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        
        if mse < best_mse:
            best_mse = mse
            best_feature = feature
            best_model = model
            
    return best_feature, best_mse, best_model

def get_best_pair(df, target_col):
    """Finds the best pair of features for predicting the target."""
    features = [c for c in df.columns if c != target_col]
    best_pair = None
    best_mse = float('inf')
    best_model = None
    
    y = df[target_col]
    
    for pair in itertools.combinations(features, 2):
        X = df[list(pair)]
        model = train_linear_model(X, y)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        
        if mse < best_mse:
            best_mse = mse
            best_pair = pair
            best_model = model
            
    return best_pair, best_mse, best_model

def calculate_mse_vs_sample_size(X, y, sizes=[20, 50, 100, 200]):
    """Calculates training and validation MSE for different sample sizes."""
    # Use a fixed validation set (e.g., 20% of data)
    X_train_full, X_val, y_train_full, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = []
    
    for size in sizes:
        if size > len(X_train_full):
            continue
            
        # Subsample training data
        X_train_sub = X_train_full[:size]
        y_train_sub = y_train_full[:size]
        
        model = train_linear_model(X_train_sub, y_train_sub)
        
        # Training MSE
        y_train_pred = model.predict(X_train_sub)
        train_mse = mean_squared_error(y_train_sub, y_train_pred)
        
        # Validation MSE
        y_val_pred = model.predict(X_val)
        val_mse = mean_squared_error(y_val, y_val_pred)
        
        results.append({
            'Sample Size': size,
            'Training MSE': train_mse,
            'Validation MSE': val_mse
        })
        
    return pd.DataFrame(results)

def get_regression_plane(model, x_range, y_range):
    """Generates a meshgrid for plotting a 3D regression plane."""
    xx, yy = np.meshgrid(x_range, y_range)
    # Flatten to predict
    X_grid = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
    # Predict
    zz = model.predict(X_grid)
    # Reshape back to grid
    zz = zz.reshape(xx.shape)
    return xx, yy, zz
