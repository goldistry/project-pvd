import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class FastMLAutoARIMA:
    """
    Ultra-fast ML approach to ARIMA with smart parameter selection
    """
    
    def __init__(self):
        self.best_order = None
        self.best_aic = np.inf
        self.model = None
        self.training_history = []
        
    def smart_grid_search(self, train_data):
        """
        Smart grid search - only test promising combinations
        """
        # Include original (5,1,0) + promising alternatives
        promising_orders = [
            (5, 1, 0),  # Original best performing
            (1, 1, 0), (1, 1, 1), (2, 1, 0), (2, 1, 1),
            (0, 1, 1), (0, 1, 2), (1, 1, 2), (3, 1, 0),
            (4, 1, 0), (1, 0, 1), (2, 0, 1)
        ]
        
        print(f"Fast Auto-ARIMA: Testing {len(promising_orders)} optimized combinations...")
        
        for i, (p, d, q) in enumerate(promising_orders):
            try:
                temp_model = ARIMA(train_data, order=(p, d, q))
                fitted_model = temp_model.fit()
                
                self.training_history.append({
                    'iteration': i,
                    'parameters': (p, d, q),
                    'aic': fitted_model.aic
                })
                
                if fitted_model.aic < self.best_aic:
                    self.best_aic = fitted_model.aic
                    self.best_order = (p, d, q)
                    self.model = fitted_model
                    
            except Exception:
                continue
        
        # Fallback to simple ARIMA if nothing works
        if self.best_order is None:
            self.best_order = (1, 1, 1)
            
        print(f"Best order: {self.best_order} (AIC: {self.best_aic:.2f})")
        return self
    


def train_ml_arima_model(train_data, test_data, window_size=60):
    """
    True ML-ARIMA: Automated parameter learning with validation
    """
    print("ML-ARIMA: Starting automated parameter optimization...")
    
    # ML approach: Use validation set for parameter selection
    val_size = 300  # Small validation set for speed
    train_subset = train_data[-2500:]  # Use recent 10 years for parameter selection
    train_part = train_subset[:-val_size]
    val_part = train_subset[-val_size:]
    
    # ML candidates: Focus on most promising parameters
    candidates = [
        (5, 1, 0),  
        (4, 1, 0), (3, 1, 0), (2, 1, 0), (1, 1, 0),
        (1, 1, 1), (2, 1, 1)
    ]
    
    # ML training: Find best parameters via validation
    best_mse = np.inf
    best_order = (5, 1, 0)
    training_log = []
    
    print(f"ML-ARIMA: Training on {len(candidates)} parameter combinations...")
    
    for i, (p, d, q) in enumerate(candidates):
        try:
            # Train model
            model = ARIMA(train_part, order=(p, d, q))
            fitted = model.fit()
            
            # Validate performance
            val_pred = fitted.forecast(steps=len(val_part))
            mse = mean_squared_error(val_part, val_pred)
            
            # Log training progress (ML-style)
            training_log.append({
                'epoch': i + 1,
                'parameters': (p, d, q),
                'validation_mse': mse,
                'aic': fitted.aic
            })
            
            print(f"  Epoch {i+1}/{len(candidates)}: {(p,d,q)} -> Val MSE: {mse:.2f}")
            
            # Update best model (ML learning)
            if mse < best_mse:
                best_mse = mse
                best_order = (p, d, q)
            print(f"  ✓ New best model: {best_order} (MSE: {best_mse:.2f})")
                
        except Exception as e:
            print(f"  X Failed: {(p,d,q)}")
            continue
    
    print(f"\nML-ARIMA Training Complete!")
    print(f"Best parameters: {best_order} (Validation MSE: {best_mse:.2f})")
    
    # Use learned parameters for prediction
    history = [x for x in train_data]
    predictions = []
    errors = []
    
    test_actual = test_data[window_size:]
    total_predictions = len(test_actual)
    
    print(f"ML-ARIMA: Making {total_predictions} predictions with learned parameters...")
    
    for t in range(total_predictions):
        try:
            model = ARIMA(history[-200:], order=best_order)
            model_fit = model.fit()
            pred = model_fit.forecast()[0]
            predictions.append(pred)
            history.append(test_actual.iloc[t])
            
            if (t + 1) % 200 == 0:
                progress = (t + 1) / total_predictions * 100
                print(f"  Prediction progress: {t+1}/{total_predictions} ({progress:.1f}%)")
                
        except Exception:
            errors.append(t)
            predictions.append(history[-1])
            history.append(test_actual.iloc[t])
    
    print(f"ML-ARIMA completed! Errors: {len(errors)}/{total_predictions}")
    return np.array(predictions), errors, pd.DataFrame(training_log)