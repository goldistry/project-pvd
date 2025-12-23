import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

def calculate_metrics(actual, predicted):
    """
    Menghitung metrik evaluasi komprehensif
    
    Parameters:
    -----------
    actual : array-like
        Nilai aktual
    predicted : array-like
        Nilai prediksi
        
    Returns:
    --------
    dict: Dictionary berisi semua metrik evaluasi
    """
    # Konversi ke numpy array
    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()
    
    # Hitung metrik dasar
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    r2 = r2_score(actual, predicted)
    
    # Directional Accuracy
    actual_direction = np.diff(actual) > 0
    pred_direction = np.diff(predicted) > 0
    directional_acc = np.mean(actual_direction == pred_direction) * 100
    
    # Residuals
    residuals = actual - predicted
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'DA': directional_acc,
        'Residual_Mean': residual_mean,
        'Residual_Std': residual_std
    }

def get_best_model(metrics_dict):
    """
    Menentukan model terbaik berdasarkan voting dari semua metrik
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary dengan key = model name, value = metrics dict
        
    Returns:
    --------
    best_model : str
        Nama model terbaik
    wins : dict
        Jumlah menang per model
    """
    wins = {model: 0 for model in metrics_dict.keys()}
    
    # MAPE, RMSE, MAE: lower is better
    for metric in ['MAPE', 'RMSE', 'MAE']:
        values = {model: metrics_dict[model][metric] for model in metrics_dict}
        winner = min(values, key=values.get)
        wins[winner] += 1
    
    # R2, DA: higher is better
    for metric in ['R2', 'DA']:
        values = {model: metrics_dict[model][metric] for model in metrics_dict}
        winner = max(values, key=values.get)
        wins[winner] += 1
    
    best_model = max(wins, key=wins.get)
    
    return best_model, wins

def format_metric_display(value, metric_type):
    """
    Format nilai metrik untuk display yang konsisten
    
    Parameters:
    -----------
    value : float
        Nilai metrik
    metric_type : str
        Tipe metrik (RMSE, MAE, MAPE, R2, DA)
        
    Returns:
    --------
    str: Formatted string
    """
    if metric_type in ['RMSE', 'MAE']:
        return f"${value:.2f}"
    elif metric_type == 'MAPE':
        return f"{value:.3f}%"
    elif metric_type == 'R2':
        return f"{value:.4f}"
    elif metric_type == 'DA':
        return f"{value:.2f}%"
    else:
        return f"{value:.4f}"