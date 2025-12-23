import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from datetime import timedelta
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Laporan UAS PVD - Stock Analysis",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap;
        background-color: #ffffff; border-radius: 4px 4px 0px 0px;
        box-shadow: 0px 2px 2px #ddd;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #4e8cff; color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNGSI UTILITY
# ==========================================
@st.cache_data
def load_data(ticker, start_year):
    start_date = f"{start_year}-01-01"
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return pd.DataFrame()

# ==========================================
# 3. SIDEBAR
# ==========================================
with st.sidebar:
    st.header("⚙️ Pengaturan Data")
    ticker_input = st.text_input("Simbol Saham", value="^NYA")
    start_year = st.number_input("Tahun Mulai", value=1965)
    
    st.divider()
    st.subheader("Parameter Model")
    test_days = st.slider("Jumlah Data Test (Hari)", 500, 2000, 1250)
    window_size = st.slider("Window Size (DL)", 30, 90, 60)
    
    st.info("Aplikasi ini membandingkan ARIMA, LSTM, dan GRU secara real-time.")

# ==========================================
# 4. LOAD DATA & TABS
# ==========================================
st.title("📊 Analisis & Prediksi Pasar Saham Global")
st.markdown(f"**Studi Kasus:** {ticker_input} (NYSE Composite Index)")

df = load_data(ticker_input, start_year)

if df.empty:
    st.stop()

# Tab Structure
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Data Overview", 
    "🔍 EDA Mendalam", 
    "🧪 Training Model", 
    "📊 Evaluasi",
    "🚀 Prediksi Future"
])

# --- TAB 1: DATA OVERVIEW ---
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Start Date", df.index[0].strftime('%Y-%m-%d'))
    col3.metric("Last Close", f"${df['Close'].iloc[-1]:,.2f}")
    col4.metric("Volatility", f"{df['Close'].pct_change().std():.4f}")

    with st.expander("Lihat Dataframe"):
        st.dataframe(df.tail(10), use_container_width=True)

# --- TAB 2: EDA MENDALAM (DIPERLENGKAP) ---
with tab2:
    st.header("Exploratory Data Analysis (EDA)")
    
    # 1. Trend Analysis
    st.subheader("1. Tren & Moving Average")
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df['Close'], label='Close Price', alpha=0.6)
    ax.plot(df.index, df['MA50'], label='MA 50', linestyle='--')
    ax.plot(df.index, df['MA200'], label='MA 200', linestyle='--', color='red')
    ax.set_title("Pergerakan Harga Jangka Panjang")
    ax.legend()
    st.pyplot(fig)

    # 2. Seasonality (Heatmap)
    st.subheader("2. Analisis Musiman (Seasonality)")
    df_heat = df.copy()
    df_heat['Year'] = df_heat.index.year
    df_heat['Month'] = df_heat.index.month
    df_heat['Return'] = df_heat['Close'].pct_change()
    
    pivot_table = df_heat.pivot_table(values='Return', index='Year', columns='Month', aggfunc='mean')
    fig_heat, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_table, cmap='RdYlGn', center=0, ax=ax)
    st.pyplot(fig_heat)
    st.caption("Warna acak menandakan **tidak ada musiman yang kuat**, sehingga SARIMA tidak diperlukan.")

    # 3. Stationarity & ACF/PACF (INI YANG KURANG SEBELUMNYA)
    st.subheader("3. Uji Stasioneritas & Autokorelasi")
    col_adf1, col_adf2 = st.columns(2)
    
    with col_adf1:
        st.write("**ADF Test (Data Asli):**")
        res = adfuller(df['Close'].dropna())
        st.write(f"P-Value: {res[1]:.4f}")
        if res[1] > 0.05:
            st.error("Data Tidak Stasioner (Perlu Differencing)")
        else:
            st.success("Data Stasioner")
            
    with col_adf2:
        st.write("**ADF Test (Differencing d=1):**")
        res_diff = adfuller(df['Close'].diff().dropna())
        st.write(f"P-Value: {res_diff[1]:.4f}")
        if res_diff[1] < 0.05:
            st.success("Stasioner setelah d=1 (Siap untuk ARIMA)")

    # ACF PACF Plots
    fig_corr, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(df['Close'].diff().dropna(), lags=40, ax=ax1)
    ax1.set_title("ACF (Menentukan q)")
    plot_pacf(df['Close'].diff().dropna(), lags=40, ax=ax2)
    ax2.set_title("PACF (Menentukan p)")
    plt.tight_layout()
    st.pyplot(fig_corr)
    st.caption("Grafik ACF/PACF ini memvalidasi pemilihan parameter ARIMA(5,1,0).")

# --- TAB 3: TRAINING MODEL ---
with tab3:
    st.header("Training Model: ARIMA vs LSTM vs GRU")
    
    if st.button("Mulai Training (Ini memakan waktu ±2 menit)", type="primary"):
        with st.spinner("Sedang melatih model..."):
            
            # 1. Data Prep
            dataset = df[['Close']].values
            train_size = len(dataset) - test_days
            scaler = MinMaxScaler(feature_range=(0, 1))
            
            train_raw = dataset[:train_size]
            test_raw = dataset[train_size:]
            
            # Fix Data Leakage
            train_scaled = scaler.fit_transform(train_raw)
            test_scaled = scaler.transform(test_raw)
            
            def create_window(data, window):
                X, y = [], []
                for i in range(window, len(data)):
                    X.append(data[i-window:i, 0])
                    y.append(data[i, 0])
                return np.array(X), np.array(y)
            
            X_train, y_train = create_window(train_scaled, window_size)
            X_test, y_test = create_window(test_scaled, window_size)
            X_train_dl = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            X_test_dl = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            
            # 2. LSTM
            model_lstm = Sequential([
                LSTM(50, return_sequences=True, input_shape=(window_size, 1)), Dropout(0.2),
                LSTM(50, return_sequences=False), Dropout(0.2), Dense(25), Dense(1)
            ])
            model_lstm.compile(optimizer='adam', loss='mse')
            model_lstm.fit(X_train_dl, y_train, batch_size=64, epochs=10, verbose=0)
            pred_lstm = scaler.inverse_transform(model_lstm.predict(X_test_dl, verbose=0)).flatten()
            
            # 3. GRU
            model_gru = Sequential([
                GRU(50, return_sequences=True, input_shape=(window_size, 1)), Dropout(0.2),
                GRU(50, return_sequences=False), Dropout(0.2), Dense(25), Dense(1)
            ])
            model_gru.compile(optimizer='adam', loss='mse')
            model_gru.fit(X_train_dl, y_train, batch_size=64, epochs=10, verbose=0)
            pred_gru = scaler.inverse_transform(model_gru.predict(X_test_dl, verbose=0)).flatten()
            
            # 4. ARIMA (Static Forecast utk Demo Cepat)
            history_arima = list(train_raw.flatten())
            model_arima = ARIMA(history_arima, order=(5,1,0)).fit()
            pred_arima = model_arima.forecast(steps=len(pred_lstm))
            
            # Save to Session State
            st.session_state['trained'] = True
            st.session_state['actual'] = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
            st.session_state['p_lstm'] = pred_lstm
            st.session_state['p_gru'] = pred_gru
            st.session_state['p_arima'] = pred_arima.values if hasattr(pred_arima, 'values') else pred_arima
            
        st.success("Training Selesai! Silakan cek tab Evaluasi.")

# --- TAB 4: EVALUASI ---
with tab4:
    if st.session_state.get('trained'):
        actual = st.session_state['actual']
        p_lstm = st.session_state['p_lstm']
        p_gru = st.session_state['p_gru']
        p_arima = st.session_state['p_arima']
        
        # Samakan panjang
        min_len = min(len(actual), len(p_lstm), len(p_gru), len(p_arima))
        
        def get_metrics(y_true, y_pred, name):
            rmse = np.sqrt(mean_squared_error(y_true[:min_len], y_pred[:min_len]))
            mae = mean_absolute_error(y_true[:min_len], y_pred[:min_len])
            r2 = r2_score(y_true[:min_len], y_pred[:min_len])
            return [name, rmse, mae, r2]
            
        metrics = [
            get_metrics(actual, p_arima, 'ARIMA'),
            get_metrics(actual, p_lstm, 'LSTM'),
            get_metrics(actual, p_gru, 'GRU')
        ]
        
        df_metrics = pd.DataFrame(metrics, columns=['Model', 'RMSE', 'MAE', 'R2'])
        st.dataframe(df_metrics.style.highlight_min(subset=['RMSE', 'MAE'], color='lightgreen')
                     .highlight_max(subset=['R2'], color='lightgreen'))
        
        # Plot Zoom In
        st.subheader("Visualisasi Prediksi (Zoom-in 100 Hari)")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(actual[-100:], label='Actual', color='black', linewidth=2)
        ax.plot(p_arima[-100:], label='ARIMA', linestyle='--')
        ax.plot(p_lstm[-100:], label='LSTM')
        ax.plot(p_gru[-100:], label='GRU')
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Silakan lakukan Training Model terlebih dahulu.")

# --- TAB 5: PREDIKSI MASA DEPAN (FINAL) ---
with tab5:
    st.header("Prediksi 60 Hari Ke Depan (ARIMA)")
    
    if st.button("Generate Forecast Bullish/Bearish"):
        with st.spinner("Retraining ARIMA with Full Data..."):
            
            # Retrain Full Data
            model_final = ARIMA(df['Close'], order=(5,1,0)).fit()
            
            # Forecast
            forecast_res = model_final.get_forecast(steps=60)
            pred_mean = forecast_res.predicted_mean
            conf_int = forecast_res.conf_int()
            
            # DF Construction
            last_date = df.index[-1]
            dates = pd.date_range(start=last_date + timedelta(days=1), periods=60)
            
            future_df = pd.DataFrame({
                'Forecast': pred_mean.values,
                'Lower': conf_int.iloc[:, 0].values,
                'Upper': conf_int.iloc[:, 1].values
            }, index=dates)
            
            # Bullish/Bearish Logic
            last_close = df['Close'].iloc[-1]
            target = future_df['Forecast'].iloc[-1]
            pct = ((target - last_close)/last_close)*100
            
            status = "BULLISH" if pct > 0 else "BEARISH"
            color = '#00CC66' if pct > 0 else '#FF3333'
            
            # --- VISUALISASI FINAL ---
            st.subheader(f"Status: {status} ({pct:.2f}%)")
            
            fig, ax = plt.subplots(figsize=(16, 8))
            history = df['Close'].iloc[-100:]
            
            # 1. Historical
            ax.plot(history.index, history, color='black', label='Data Historis')
            
            # 2. Connector
            ax.plot([history.index[-1], future_df.index[0]],
                    [history.iloc[-1], future_df['Forecast'].iloc[0]],
                    color=color, linestyle='--')
            
            # 3. Forecast
            ax.plot(future_df.index, future_df['Forecast'], color=color, linewidth=3, label='Prediksi')
            
            # 4. Confidence Interval
            ax.fill_between(future_df.index, future_df['Lower'], future_df['Upper'],
                            color=color, alpha=0.15, label='Confidence Interval 95%')
            
            # 5. Annotation
            ax.scatter(future_df.index[-1], target, color=color, s=100)
            ax.text(future_df.index[-1], target, f" Target:\n {target:.2f}", 
                    color=color, fontweight='bold', va='center')
            
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.success(f"Model memproyeksikan tren **{status}**. Area arsiran menunjukkan ketidakpastian statistik.")