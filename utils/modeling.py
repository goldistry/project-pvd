import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st

def train_arima_model(train_data, test_data, order=(5, 1, 0), window_size=60):
    """
    Train ARIMA model with rolling prediction
    
    Parameters:
    -----------
    train_data : pd.Series
        Training data
    test_data : pd.Series
        Testing data
    order : tuple
        ARIMA order (p, d, q)
    window_size : int
        Window size to align with LSTM
        
    Returns:
    --------
    tuple
        (predictions, errors)
    """
    history = [x for x in train_data]
    predictions = []
    errors = []
    
    # Align test data with LSTM (skip first window_size points)
    test_actual = test_data[window_size:]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for t in range(len(test_actual)):
        try:
            # Use last 200 data points for efficiency
            model = ARIMA(history[-200:], order=order)
            model_fit = model.fit()
            pred = model_fit.forecast()[0]
            predictions.append(pred)
            history.append(test_actual.iloc[t])
            
            # Update progress
            if (t + 1) % 50 == 0:
                progress = int((t + 1) / len(test_actual) * 100)
                progress_bar.progress(progress)
                status_text.text(f"ARIMA Progress: {t+1}/{len(test_actual)} predictions")
        
        except Exception as e:
            errors.append(t)
            # Fallback: use last value
            predictions.append(history[-1])
            history.append(test_actual.iloc[t])
    
    progress_bar.empty()
    status_text.empty()
    
    return np.array(predictions), errors

def build_lstm_model(window_size, units=50, dropout=0.2):
    """
    Build LSTM model architecture
    
    Parameters:
    -----------
    window_size : int
        Number of time steps
    units : int
        Number of LSTM units
    dropout : float
        Dropout rate
        
    Returns:
    --------
    keras.Model
        Compiled LSTM model
    """
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(window_size, 1)),
        Dropout(dropout),
        LSTM(units, return_sequences=False),
        Dropout(dropout),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_gru_model(window_size, units=50, dropout=0.2):
    """
    Build GRU model architecture
    
    Parameters:
    -----------
    window_size : int
        Number of time steps
    units : int
        Number of GRU units
    dropout : float
        Dropout rate
        
    Returns:
    --------
    keras.Model
        Compiled GRU model
    """
    model = Sequential([
        GRU(units, return_sequences=True, input_shape=(window_size, 1)),
        Dropout(dropout),
        GRU(units, return_sequences=False),
        Dropout(dropout),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_deep_learning_model(model, X_train, y_train, epochs=50, batch_size=32, model_name="Model"):
    """
    Train LSTM or GRU model
    
    Parameters:
    -----------
    model : keras.Model
        Model to train
    X_train : np.array
        Training features
    y_train : np.array
        Training labels
    epochs : int
        Number of epochs
    batch_size : int
        Batch size
    model_name : str
        Name for progress display
        
    Returns:
    --------
    History
        Training history
    """
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=0
    )
    
    # Custom callback for progress
    class ProgressCallback:
        def __init__(self, total_epochs):
            self.total_epochs = total_epochs
            self.progress_bar = st.progress(0)
            self.status_text = st.empty()
        
        def on_epoch_end(self, epoch):
            progress = int((epoch + 1) / self.total_epochs * 100)
            self.progress_bar.progress(progress)
            self.status_text.text(f"{model_name} Training: Epoch {epoch + 1}/{self.total_epochs}")
        
        def cleanup(self):
            self.progress_bar.empty()
            self.status_text.empty()
    
    progress_callback = ProgressCallback(epochs)
    
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Update progress for each epoch manually
    for epoch in range(len(history.history['loss'])):
        progress_callback.on_epoch_end(epoch)
    
    progress_callback.cleanup()
    
    return history

def predict_future_arima(data, steps=60, order=(5, 1, 0)):
    """
    Predict future values using ARIMA
    
    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Full historical data
    steps : int
        Number of steps to forecast
    order : tuple
        ARIMA order (p, d, q)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with Forecast, Lower_Bound, Upper_Bound columns
    """
    # Extract Close column if DataFrame, ensure it's a Series
    if isinstance(data, pd.DataFrame):
        series = data['Close']
    else:
        series = data
    
    # Ensure numeric data
    series = pd.to_numeric(series, errors='coerce').dropna()
    
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    
    forecast_result = model_fit.get_forecast(steps=steps)
    forecast_values = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()
    
    # Create future dates
    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='D')
    
    # Create DataFrame
    forecast_df = pd.DataFrame({
        'Forecast': forecast_values.values,
        'Lower_Bound': conf_int.iloc[:, 0].values,
        'Upper_Bound': conf_int.iloc[:, 1].values
    }, index=future_dates)
    
    return forecast_df