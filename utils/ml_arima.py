import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
import itertools
import warnings
warnings.filterwarnings('ignore')

class MLAutoARIMA:
    """
    Machine Learning approach to ARIMA using automated parameter selection
    """
    
    def __init__(self, max_p=5, max_d=2, max_q=5):
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.best_order = None
        self.best_aic = np.inf
        self.model = None
        self.training_history = []
        
    def grid_search_fit(self, train_data, validation_data=None):
        """
        ML-style grid search to find optimal ARIMA parameters
        """
        # Create parameter grid
        p_values = range(0, self.max_p + 1)
        d_values = range(0, self.max_d + 1)
        q_values = range(0, self.max_q + 1)
        
        param_combinations = list(itertools.product(p_values, d_values, q_values))
        
        print(f"Training Auto-ARIMA with {len(param_combinations)} parameter combinations...")
        
        for i, (p, d, q) in enumerate(param_combinations):
            try:
                # Fit ARIMA model
                temp_model = ARIMA(train_data, order=(p, d, q))
                fitted_model = temp_model.fit()
                
                # Calculate validation score
                if validation_data is not None:
                    # Use validation data for model selection
                    forecast = fitted_model.forecast(steps=len(validation_data))
                    mse = mean_squared_error(validation_data, forecast)
                    score = mse
                else:
                    # Use AIC for model selection
                    score = fitted_model.aic
                
                # Store training history (ML-style)
                self.training_history.append({
                    'iteration': i,
                    'parameters': (p, d, q),
                    'score': score,
                    'aic': fitted_model.aic
                })
                
                # Update best model
                if score < self.best_aic:
                    self.best_aic = score
                    self.best_order = (p, d, q)
                    self.model = fitted_model
                    
            except Exception as e:
                continue
        
        print(f"Best ARIMA order found: {self.best_order} with score: {self.best_aic:.4f}")
        return self
    
    def predict(self, steps=1):
        """ML-style predict method"""
        if self.model is None:
            raise ValueError("Model not fitted yet!")
        return self.model.forecast(steps=steps)
    
    def get_training_history(self):
        """Get training history like ML models"""
        return pd.DataFrame(self.training_history)

def train_ml_arima_model(train_data, test_data, window_size=60):
    """
    Train ARIMA using ML approach with automated parameter selection
    """
    # Split training data for validation
    val_size = int(len(train_data) * 0.2)
    train_subset = train_data[:-val_size]
    val_subset = train_data[-val_size:]
    
    # Initialize ML Auto-ARIMA
    ml_arima = MLAutoARIMA(max_p=5, max_d=2, max_q=3)
    
    # Fit with grid search (ML approach)
    ml_arima.grid_search_fit(train_subset, val_subset)
    
    # Make rolling predictions (like original)
    history = [x for x in train_data]
    predictions = []
    errors = []
    
    test_actual = test_data[window_size:]
    
    for t in range(len(test_actual)):
        try:
            # Use the best order found by ML approach
            model = ARIMA(history[-200:], order=ml_arima.best_order)
            model_fit = model.fit()
            pred = model_fit.forecast()[0]
            predictions.append(pred)
            history.append(test_actual.iloc[t])
        except Exception as e:
            errors.append(t)
            predictions.append(history[-1])
            history.append(test_actual.iloc[t])
    
    return np.array(predictions), errors, ml_arima.get_training_history()