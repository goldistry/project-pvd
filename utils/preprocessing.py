import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller

def calculate_technical_indicators(data):
    """
    Menambahkan indikator teknikal ke data
    """
    df = data.copy()
    
    # Moving Averages
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    
    # Daily Return
    df['Daily_Return'] = df['Close'].pct_change() * 100
    
    # Volatility (30-day rolling std of returns)
    df['Volatility_30'] = df['Daily_Return'].rolling(window=30).std()
    
    # Decade untuk analisis
    df['Decade'] = (df.index.year // 10) * 10
    
    return df

def test_stationarity(timeseries):
    """
    Melakukan Augmented Dickey-Fuller test untuk stationarity
    
    Returns:
        dict: Hasil ADF test dengan interpretasi
    """
    result = adfuller(timeseries.dropna(), autolag='AIC')
    
    is_stationary = result[1] <= 0.05
    
    interpretation = "STATIONARY" if is_stationary else "NON-STATIONARY"
    
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': is_stationary,
        'interpretation': interpretation
    }

def prepare_lstm_data(data, train_ratio=0.8, window_size=60):
    """
    Prepare data untuk LSTM/GRU training dengan proper scaling
    
    Parameters:
    -----------
    data : DataFrame
        Dataset dengan kolom 'Close'
    train_ratio : float
        Rasio data training
    window_size : int
        Lookback period untuk sequences
        
    Returns:
    --------
    X_train, y_train, X_test, y_test, scaler, train_size
    """
    dataset = data[['Close']].values
    
    # Split data
    train_size = int(len(dataset) * train_ratio)
    train_raw = dataset[:train_size]
    test_raw = dataset[train_size:]
    
    # Fit scaler HANYA pada training data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled = scaler.transform(test_raw)
    
    # Create sequences
    def create_sequences(data, window):
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i-window:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    X_train, y_train = create_sequences(train_scaled, window_size)
    X_test, y_test = create_sequences(test_scaled, window_size)
    
    # Reshape untuk LSTM input: (samples, time_steps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    return X_train, y_train, X_test, y_test, scaler, train_size

def prepare_arima_data(data, train_ratio=0.8):
    """
    Prepare data untuk ARIMA (tidak perlu scaling)
    
    Returns:
    --------
    train_data, test_data, train_size
    """
    train_size = int(len(data) * train_ratio)
    
    train_data = data['Close'][:train_size]
    test_data = data['Close'][train_size:]
    
    return train_data, test_data, train_size