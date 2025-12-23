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

def prepare_lstm_data(data, window_size=60, test_years=5):
    """
    Prepare data untuk LSTM/GRU training dengan proper scaling
    Use all historical data for training, recent years for testing
    
    Parameters:
    -----------
    data : DataFrame
        Dataset dengan kolom 'Close'
    window_size : int
        Lookback period untuk sequences
    test_years : int
        Years of recent data to use for testing
        
    Returns:
    --------
    X_train, y_train, X_test, y_test, scaler, train_size
    """
    # Ensure parameters are integers
    window_size = int(window_size)
    test_years = int(test_years)
    
    # Use last 5 years for testing, rest for training
    test_points = test_years * 250
    
    if len(data) > test_points:
        train_data = data.iloc[:-test_points]
        test_data = data.iloc[-test_points:]
        print(f"Training: {len(train_data)} points (~{len(train_data)/250:.1f} years)")
        print(f"Testing: {len(test_data)} points (~{len(test_data)/250:.1f} years)")
    else:
        # Fallback if not enough data
        train_size = int(len(data) * 0.8)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        print(f"Fallback split - Training: {len(train_data)}, Testing: {len(test_data)}")
    
    train_dataset = train_data[['Close']].values
    test_dataset = test_data[['Close']].values
    
    # Fit scaler HANYA pada training data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_dataset)
    test_scaled = scaler.transform(test_dataset)
    
    # Create sequences
    def create_sequences(data, window):
        window = int(window)  # Ensure window is integer
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
    
    return X_train, y_train, X_test, y_test, scaler, len(train_data)

def prepare_arima_data(data, test_years=5):
    """
    Prepare data untuk ARIMA (tidak perlu scaling)
    Use all historical data for training, recent years for testing
    
    Returns:
    --------
    train_data, test_data, train_size
    """
    # Ensure parameter is integer
    test_years = int(test_years)
    
    # Use last 5 years for testing, rest for training
    test_points = test_years * 250
    
    if len(data) > test_points:
        train_data = data['Close'].iloc[:-test_points]
        test_data = data['Close'].iloc[-test_points:]
        print(f"ARIMA Training: {len(train_data)} points (~{len(train_data)/250:.1f} years)")
        print(f"ARIMA Testing: {len(test_data)} points (~{len(test_data)/250:.1f} years)")
    else:
        # Fallback if not enough data
        train_size = int(len(data) * 0.8)
        train_data = data['Close'].iloc[:train_size]
        test_data = data['Close'].iloc[train_size:]
        print(f"ARIMA Fallback - Training: {len(train_data)}, Testing: {len(test_data)}")
    
    return train_data, test_data, len(train_data)