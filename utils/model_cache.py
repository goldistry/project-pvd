import pickle
import os
import pandas as pd
import numpy as np
from utils.data_loader import load_data, filter_index_data
from utils.preprocessing import calculate_technical_indicators, prepare_lstm_data, prepare_arima_data
from utils.modeling import build_lstm_model, build_gru_model, train_deep_learning_model
from utils.ml_arima import train_ml_arima_model
from utils.metrics import calculate_metrics

CACHE_DIR = "model_cache"

def ensure_cache_dir():
    """Create cache directory if it doesn't exist"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def get_cache_path(index_name, model_type):
    """Get cache file path for specific index and model"""
    return os.path.join(CACHE_DIR, f"{index_name}_{model_type}.pkl")

def save_model_results(index_name, results):
    """Save all model results to cache"""
    ensure_cache_dir()
    cache_path = os.path.join(CACHE_DIR, f"{index_name}_results.pkl")
    with open(cache_path, 'wb') as f:
        pickle.dump(results, f)

def load_model_results(index_name):
    """Load cached model results"""
    cache_path = os.path.join(CACHE_DIR, f"{index_name}_results.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None

def is_cache_valid(index_name):
    """Check if cache exists and is valid"""
    cache_path = os.path.join(CACHE_DIR, f"{index_name}_results.pkl")
    return os.path.exists(cache_path)

def train_and_cache_models(index_name='NYA', train_ratio=0.8, window_size=60, epochs=50):
    """Train all models and cache results"""
    print(f"Training models for {index_name}...")
    
    # Load and prepare data
    df = load_data()
    data = filter_index_data(df, index_name)
    data = calculate_technical_indicators(data)
    data['Close_Diff'] = data['Close'].diff()
    
    # Prepare data for models
    X_train, y_train, X_test, y_test, scaler, train_size = prepare_lstm_data(
        data, window_size
    )
    train_data_arima, test_data_arima, _ = prepare_arima_data(data)
    
    # Train ML-ARIMA
    print("Training ML-ARIMA...")
    arima_preds, arima_errors, _ = train_ml_arima_model(
        train_data_arima, test_data_arima, window_size=window_size
    )
    
    # Train LSTM
    print("Training LSTM...")
    model_lstm = build_lstm_model(window_size)
    history_lstm = train_deep_learning_model(
        model_lstm, X_train, y_train, epochs=epochs, model_name="LSTM"
    )
    lstm_preds_scaled = model_lstm.predict(X_test, verbose=0)
    lstm_preds = scaler.inverse_transform(lstm_preds_scaled).flatten()
    
    # Train GRU
    print("Training GRU...")
    model_gru = build_gru_model(window_size)
    history_gru = train_deep_learning_model(
        model_gru, X_train, y_train, epochs=epochs, model_name="GRU"
    )
    gru_preds_scaled = model_gru.predict(X_test, verbose=0)
    gru_preds = scaler.inverse_transform(gru_preds_scaled).flatten()
    
    # Calculate metrics
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    metrics_arima = calculate_metrics(actual_prices, arima_preds)
    metrics_lstm = calculate_metrics(actual_prices, lstm_preds)
    metrics_gru = calculate_metrics(actual_prices, gru_preds)
    
    # Prepare results
    results = {
        'data': data,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'train_size': train_size,
        'test_data_arima': test_data_arima,
        'arima_preds': arima_preds,
        'lstm_preds': lstm_preds,
        'gru_preds': gru_preds,
        'actual_prices': actual_prices,
        'metrics_arima': metrics_arima,
        'metrics_lstm': metrics_lstm,
        'metrics_gru': metrics_gru,
        'history_lstm': history_lstm,
        'history_gru': history_gru,
        'model_lstm': model_lstm,
        'model_gru': model_gru,
        'selected_index': index_name
    }
    
    # Save to cache
    save_model_results(index_name, results)
    print(f"Models trained and cached for {index_name}")
    
    return results

def get_or_train_models(index_name='NYA', force_retrain=False):
    """Get cached models or train if not available"""
    if not force_retrain and is_cache_valid(index_name):
        print(f"Loading cached models for {index_name}...")
        return load_model_results(index_name)
    else:
        return train_and_cache_models(index_name)

if __name__ == "__main__":
    # Pre-train models for common indices
    indices = ['NYA', 'IXIC', 'DJI', 'GSPC']
    
    for index in indices:
        try:
            train_and_cache_models(index)
        except Exception as e:
            print(f"Error training {index}: {e}")
            continue
    
    print("All models pre-trained and cached!")