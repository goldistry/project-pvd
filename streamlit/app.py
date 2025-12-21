import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Stock Market Analysis - ARIMA vs LSTM",
    page_icon="📈",
    layout="wide"
)

st.title("Analisis Prediksi Stock Market: ARIMA vs LSTM")
st.markdown("""
**Tugas Ujian Akhir - Pemrograman dan Visualisasi Data**

Aplikasi ini membandingkan performa **ARIMA (Statistical Model)** dengan **LSTM (Deep Learning)** 
dalam memprediksi harga penutupan saham NYSE Index (NYA).
""")

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("Pengaturan")
st.sidebar.markdown("---")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('../dataset/indexProcessed.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        st.error("File dataset tidak ditemukan!")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# Select index
available_indices = sorted(df['Index'].unique())
selected_index = st.sidebar.selectbox(
    "Pilih Stock Index:",
    available_indices,
    index=available_indices.index('NYA') if 'NYA' in available_indices else 0
)

# Filter data
data = df[df['Index'] == selected_index].sort_values('Date').reset_index(drop=True)
data.set_index('Date', inplace=True)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.info(f"""
**Dataset Info:**
- Index: {selected_index}
- Total Data: {len(data)} hari
- Periode: {data.index.min().date()} s/d {data.index.max().date()}
- Durasi: {(data.index.max() - data.index.min()).days / 365.25:.1f} tahun
""")

# Model settings
st.sidebar.markdown("---")
st.sidebar.subheader("Model Settings")
train_ratio = st.sidebar.slider("Training Data Ratio (%)", 60, 90, 80, 5) / 100
window_size = st.sidebar.slider("LSTM Window Size", 30, 120, 60, 10)
lstm_epochs = st.sidebar.slider("LSTM Epochs", 20, 100, 50, 10)

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview & EDA",
    "Stationarity Test",
    "Model Training",
    "Model Comparison"
])

# ==========================================
# TAB 1: OVERVIEW & EDA
# ==========================================
with tab1:
    st.header("Exploratory Data Analysis")
    
    # === 1. DATASET OVERVIEW ===
    st.subheader("1. Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Data Points", f"{len(data):,}")
    col2.metric("Latest Close Price", f"${data['Close'].iloc[-1]:,.2f}")
    col3.metric("Highest Price", f"${data['Close'].max():,.2f}")
    col4.metric("Lowest Price", f"${data['Close'].min():,.2f}")
    
    with st.expander("Preview Data"):
        st.dataframe(data.head(10), use_container_width=True)
    
    with st.expander("Statistik Deskriptif"):
        st.dataframe(data.describe(), use_container_width=True)
    
    st.markdown("---")
    
    # === 2. TIME SERIES PLOT ===
    st.subheader("2. Tren Harga dengan Moving Average")
    
    # Calculate MAs
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['MA_200'] = data['Close'].rolling(window=200).mean()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Price with MA
    ax1.plot(data.index, data['Close'], label='Close Price', color='#1f77b4', linewidth=1.5, alpha=0.8)
    ax1.plot(data.index, data['MA_50'], label='MA-50', color='#2ca02c', linewidth=2, linestyle='--')
    ax1.plot(data.index, data['MA_200'], label='MA-200', color='#d62728', linewidth=2, linestyle='--')
    ax1.set_title(f'Price Trend: {selected_index}', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Volume
    ax2.bar(data.index, data['Volume'], color='#9467bd', alpha=0.6, width=10)
    ax2.set_title('Trading Volume', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volume')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    with st.expander("Penjelasan Visualisasi: Time Series Plot"):
        st.markdown("""
        **Mengapa jenis plot ini?**
        - Line plot adalah pilihan terbaik untuk time series data karena menunjukkan kontinuitas temporal dan tren over time
        - Moving Average menghaluskan noise dan menunjukkan tren jangka pendek (MA-50) vs jangka panjang (MA-200)
        - Bar chart untuk volume memudahkan identifikasi periode trading aktif dan konfirmasi kekuatan trend
        
        **Aesthetics & Mapping:**
        - **X-axis (Position):** Date (Temporal/Continuous) - waktu dipetakan ke posisi horizontal
        - **Y-axis (Position):** Price & Volume (Continuous) - nilai dipetakan ke posisi vertikal
        - **Color (Hue):** Variable Type (Discrete/Categorical) - setiap variabel diberi warna berbeda
        - **Line Style:** Solid untuk actual price, dashed untuk moving averages
        
        **Pemilihan Warna (Multiple Discrete Hues):**
        - **Biru (#1f77b4) = Close Price:** Warna utama karena Close adalah harga paling penting untuk analisis
        - **Hijau (#2ca02c) = MA-50:** Trend jangka pendek, hijau mengindikasikan growth signal
        - **Merah (#d62728) = MA-200:** Trend jangka panjang, merah lebih kontras untuk pembeda signifikan
        - **Ungu (#9467bd) = Volume:** Warna terpisah untuk data berbeda skala
        
        **Mengapa discrete color?** Karena membandingkan kategori berbeda (Close vs MA-50 vs MA-200 vs Volume), bukan continuous gradient.
        
        **Interpretasi:**
        - **Golden Cross:** Ketika MA-50 memotong MA-200 ke atas menunjukkan bullish signal (momentum naik)
        - **Death Cross:** Ketika MA-50 memotong MA-200 ke bawah menunjukkan bearish signal (momentum turun)
        - **Volume Spike:** Trading volume tinggi yang diikuti pergerakan harga signifikan mengkonfirmasi kekuatan trend
        - Dari grafik terlihat tren naik jangka panjang pada NYA index dengan beberapa periode koreksi
        """)
    
    st.markdown("---")
    
    # === 3. DISTRIBUTION ANALYSIS ===
    st.subheader("3. Distribusi Daily Return")
    
    data['Daily_Return'] = data['Close'].pct_change() * 100
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    
    # Histogram Daily Return
    ax.hist(data['Daily_Return'].dropna(), bins=100, color='#2ca02c', alpha=0.6, edgecolor='black', density=True)
    data['Daily_Return'].plot(kind='kde', ax=ax, color='#d62728', linewidth=2, label='KDE')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_title('Distribusi Daily Return', fontsize=12, fontweight='bold')
    ax.set_xlabel('Daily Return (%)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("Mean Return", f"{data['Daily_Return'].mean():.4f}%")
        st.metric("Std Deviation", f"{data['Daily_Return'].std():.4f}%")
    with col_stat2:
        st.metric("Skewness", f"{data['Daily_Return'].skew():.4f}")
        st.metric("Kurtosis", f"{data['Daily_Return'].kurtosis():.4f}")
    
    with st.expander("Penjelasan Visualisasi: Distribution Analysis"):
        st.markdown("""
        **Mengapa jenis plot ini?**
        - **Histogram** menunjukkan frekuensi distribusi data return secara diskrit, penting untuk memahami risk profile
        - **KDE (Kernel Density Estimation)** memberikan smooth curve untuk melihat bentuk distribusi secara keseluruhan
        - Kombinasi keduanya memberikan pemahaman komprehensif tentang karakteristik return
        
        **Aesthetics & Mapping:**
        - **X-axis:** Return value dalam persen (Continuous)
        - **Y-axis:** Density (Continuous, normalized frequency)
        - **Bars:** Histogram bins dengan height proporsional ke frequency
        - **Line:** KDE smooth curve untuk distribusi continuous
        - **Vertical line (x=0):** Reference point untuk zero return
        - **Transparency (alpha=0.6):** Untuk layering histogram dan KDE
        
        **Pemilihan Warna:**
        - **Hijau (histogram bars):** Discrete color untuk return distribution
        - **Merah (KDE line):** Kontras tinggi untuk visibility, konsisten di seluruh visualisasi
        - **Bukan continuous scale** karena tidak ada gradasi nilai yang perlu ditunjukkan
        
        **Interpretasi:**
        - **Mean Return positif:** Secara rata-rata index cenderung naik (return > 0)
        - **Skewness negatif:** Distribusi condong kiri, menunjukkan lebih banyak extreme downside events (crash) dibanding upside
        - **High Kurtosis (> 3):** Fat tails menunjukkan extreme events (big gains/losses) lebih sering terjadi dari normal distribution
        - **Implikasi untuk modeling:** Data memiliki karakteristik non-normal dengan tail risk, penting untuk model yang robust terhadap outliers
        """)
    
    st.markdown("---")
    
    # === 4. CORRELATION HEATMAP ===
    st.subheader("4. Correlation Matrix")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Select numeric columns only
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    corr_matrix = data[numeric_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0, vmin=-1, vmax=1,
                fmt='.2f', linewidths=0.5, square=True, cbar_kws={'label': 'Correlation'},
                ax=ax)
    ax.set_title('Correlation Heatmap', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    
    with st.expander("Penjelasan Visualisasi: Correlation Heatmap"):
        st.markdown("""
        **Mengapa jenis plot ini?**
        - Heatmap ideal untuk visualisasi correlation matrix karena menggunakan color intensity untuk encode nilai korelasi
        - Dapat menampilkan banyak variabel sekaligus dalam satu view yang compact
        - Mudah mengidentifikasi pola korelasi secara visual tanpa membaca angka satu per satu
        
        **Aesthetics & Mapping:**
        - **Rows & Columns (Position):** Feature names (Categorical) - setiap cell merepresentasikan korelasi 2 fitur
        - **Cell Color (Intensity):** Correlation coefficient (Continuous: -1 to +1)
        - **Annotations (Text):** Exact correlation values di setiap cell untuk precision
        - **Square cells:** Aspect ratio 1:1 untuk visual symmetry
        
        **Pemilihan Warna (Continuous Diverging):**
        - **Colormap: RdYlGn (Red-Yellow-Green)** - Diverging colormap dengan 3 regions:
          - RED (-1.0 to -0.3): Korelasi negatif kuat
          - YELLOW (-0.3 to +0.3): Korelasi lemah atau tidak ada
          - GREEN (+0.3 to +1.0): Korelasi positif kuat
        - **Center at 0 (yellow):** Menunjukkan "no correlation" sebagai neutral point
        - **Continuous scale:** Karena correlation adalah nilai continuous, bukan diskrit
        - **Double hue (diverging):** Untuk membedakan arah korelasi (positif vs negatif)
        
        **Interpretasi:**
        - **Open-High-Low-Close-AdjClose:** Korelasi sangat tinggi (>0.99) - EXPECTED karena semua merupakan harga pada hari yang sama
        - **Volume:** Korelasi rendah/sedang dengan price features - menunjukkan volume memberikan informasi independen
        - **High correlation between price features:** Menunjukkan redundancy, untuk modeling cukup gunakan satu (Close)
        - **Implikasi untuk modeling:** Tidak perlu semua price features, fokus pada Close dan Volume untuk menghindari multicollinearity
        """)
    
    st.markdown("---")
    
    # === 5. VOLATILITY ANALYSIS ===
    st.subheader("5. Volatility Analysis")
    
    data['Volatility_30'] = data['Daily_Return'].rolling(window=30).std()
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    
    # Rolling Volatility
    ax.plot(data.index, data['Volatility_30'], color='#E63946', linewidth=1.5)
    ax.set_title('Rolling Volatility (30-Day)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility (%)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    with st.expander("Penjelasan Visualisasi: Volatility Analysis"):
        st.markdown("""
        **Mengapa jenis plot ini?**
        - Line plot menunjukkan perubahan volatilitas over time - critical untuk understanding risk dynamics
        - Rolling window (30-day) menghaluskan fluktuasi jangka pendek untuk melihat trend volatility
        
        **Aesthetics & Mapping:**
        - **X-axis:** Date (Temporal/Sequential)
        - **Y-axis:** Standard deviation of returns (Continuous, dalam %)
        - **Line color:** Red (#E63946) - single hue untuk emphasis
        - **Line width:** 1.5px untuk visibility
        
        **Pemilihan Warna:**
        - **Merah untuk volatility:** Konvensi universal bahwa merah = warning/risk/danger
        - **Single discrete hue** karena hanya satu metric yang divisualisasikan
        
        **Interpretasi:**
        - **Spike pada volatility:** Mengindikasikan periode market uncertainty (crisis, major news, earnings surprises)
        - **Low volatility periods:** Market stability, cenderung tren yang smooth
        - **Volatility clustering:** Periode high volatility cenderung diikuti high volatility (GARCH effect)
        - **Implikasi untuk modeling:** Period dengan high volatility lebih sulit diprediksi, model perlu robust terhadap regime changes
        - Terlihat beberapa spike volatility yang mengindikasikan events besar (financial crisis, pandemic, dll)
        """)

# ==========================================
# TAB 2: STATIONARITY TEST
# ==========================================
with tab2:
    st.header("Stationarity Test (untuk ARIMA)")
    
    st.info("""
    **Mengapa Stationarity Penting?**
    
    ARIMA model membutuhkan data yang **stationary** (mean, variance, autocorrelation konstan over time).
    Kita gunakan **Augmented Dickey-Fuller (ADF) Test** untuk menguji stationarity.
    
    - **H0 (Null Hypothesis):** Data memiliki unit root (NON-STATIONARY)
    - **H1 (Alternative):** Data tidak memiliki unit root (STATIONARY)
    - **Decision:** Jika p-value ≤ 0.05, reject H0, data STATIONARY
    """)
    
    # ADF Test function
    def adf_test(series, name):
        result = adfuller(series.dropna(), autolag='AIC')
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("ADF Statistic", f"{result[0]:.6f}")
            st.metric("p-value", f"{result[1]:.6f}")
            
            if result[1] <= 0.05:
                st.success("STATIONARY")
            else:
                st.error("NON-STATIONARY")
        
        with col2:
            st.write("**Critical Values:**")
            for key, value in result[4].items():
                st.write(f"- {key}: {value:.3f}")
            
            if result[1] <= 0.05:
                st.write("p-value ≤ 0.05: Reject H0, Data STATIONARY")
            else:
                st.write("p-value > 0.05: Gagal reject H0, Data NON-STATIONARY")
        
        return result
    
    st.subheader("1. Test pada Level (Original Data)")
    result_level = adf_test(data['Close'], "Close Price (Level)")
    
    st.markdown("---")
    
    st.subheader("2. Test pada First Difference")
    data['Close_Diff'] = data['Close'].diff()
    result_diff = adf_test(data['Close_Diff'], "Close Price (Differenced)")
    
    st.markdown("---")
    
    # ACF & PACF Plots
    st.subheader("3. ACF & PACF Plots (Untuk Menentukan Parameter ARIMA)")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    plot_acf(data['Close_Diff'].dropna(), lags=40, ax=ax1)
    ax1.set_title('ACF - Untuk Menentukan MA Parameter (q)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Correlation')
    
    plot_pacf(data['Close_Diff'].dropna(), lags=40, ax=ax2)
    ax2.set_title('PACF - Untuk Menentukan AR Parameter (p)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Correlation')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    with st.expander("Penjelasan Visualisasi: ACF & PACF"):
        st.markdown("""
        **Mengapa jenis plot ini?**
        - **ACF (Autocorrelation Function):** Mengukur korelasi antara time series dengan lag-nya, essential untuk time series analysis
        - **PACF (Partial ACF):** Mengukur korelasi setelah menghilangkan efek lag intermediate
        - Tool standar untuk menentukan parameter optimal ARIMA (p, d, q)
        
        **Aesthetics & Mapping:**
        - **X-axis (Position):** Lag number (Discrete: 0, 1, 2, ..., 40)
        - **Y-axis (Position):** Correlation coefficient (Continuous: -1 to 1)
        - **Vertical lines (stems):** Correlation value at each lag
        - **Blue shaded area:** 95% confidence interval
        - **Bars outside shaded area:** Statistically significant correlations
        
        **Pemilihan Warna:**
        - **Blue bars:** Default matplotlib, discrete values
        - **Blue shaded region:** Continuous confidence band
        - Menggunakan single color scheme untuk consistency
        
        **Interpretasi:**
        - **ACF:** Lag yang signifikan (keluar dari blue area) menentukan parameter **q** (MA order)
        - **PACF:** Lag yang signifikan menentukan parameter **p** (AR order)
        - **Dari grafik:** PACF menunjukkan significant lags di awal (p=5), ACF decay gradually (q=0)
        - **Recommended ARIMA order:** (5,1,0) berdasarkan PACF cutoff dan ADF test result
        - Pola ini menunjukkan data memiliki autoregressive component yang kuat
        """)
    
    st.markdown("---")
    
    st.success(f"""
    **Kesimpulan Stationarity Test:**
    
    - **Original data:** {'STATIONARY' if result_level[1] <= 0.05 else 'NON-STATIONARY'}
    - **After differencing:** {'STATIONARY' if result_diff[1] <= 0.05 else 'NON-STATIONARY'}
    - **Recommended d parameter:** {0 if result_level[1] <= 0.05 else 1}
    
    Untuk modeling ARIMA, kita akan gunakan **d={0 if result_level[1] <= 0.05 else 1}** (differencing order).
    Parameter p dan q ditentukan dari ACF/PACF plots: **(p=5, d=1, q=0)**.
    """)

# ==========================================
# TAB 3: MODEL TRAINING
# ==========================================
with tab3:
    st.header("Model Training: ARIMA vs LSTM")
    
    if st.button("Train Both Models", type="primary"):
        with st.spinner("Training models... This may take a few minutes..."):
            
            # ===== DATA PREPARATION =====
            train_size = int(len(data) * train_ratio)
            
            # For ARIMA (original scale)
            train_data_arima = data['Close'][:train_size]
            test_data_arima = data['Close'][train_size:]
            
            # For LSTM (scaled)
            dataset_lstm = data[['Close']].values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(dataset_lstm)
            
            # Create sequences
            def create_sequences(data, window):
                X, y = [], []
                for i in range(window, len(data)):
                    X.append(data[i-window:i, 0])
                    y.append(data[i, 0])
                return np.array(X), np.array(y)
            
            train_scaled = scaled_data[:train_size]
            test_scaled = scaled_data[train_size:]
            
            X_train, y_train = create_sequences(train_scaled, window_size)
            X_test, y_test = create_sequences(test_scaled, window_size)
            
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # ===== TRAIN ARIMA =====
            status_text.text("Training ARIMA model...")
            progress_bar.progress(10)
            
            history_arima = [x for x in train_data_arima]
            arima_preds = []
            
            test_actual = test_data_arima[window_size:]
            
            for t in range(len(test_actual)):
                try:
                    model_arima = ARIMA(history_arima[-200:], order=(5, 1, 0))
                    res = model_arima.fit()
                    pred = res.forecast()[0]
                    arima_preds.append(pred)
                    history_arima.append(test_actual.iloc[t])
                    
                    if (t + 1) % 50 == 0:
                        progress_bar.progress(10 + int((t / len(test_actual)) * 30))
                except:
                    arima_preds.append(history_arima[-1])
                    history_arima.append(test_actual.iloc[t])
            
            progress_bar.progress(40)
            
            # ===== TRAIN LSTM =====
            status_text.text("Training LSTM model...")
            
            model_lstm = Sequential([
                LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model_lstm.compile(optimizer='adam', loss='mean_squared_error')
            
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
            
            history_lstm = model_lstm.fit(
                X_train, y_train,
                batch_size=32,
                epochs=lstm_epochs,
                validation_split=0.2,
                callbacks=[early_stop],
                verbose=0
            )
            
            progress_bar.progress(80)
            
            # ===== PREDICTIONS =====
            status_text.text("Generating predictions...")
            
            lstm_preds_scaled = model_lstm.predict(X_test, verbose=0)
            lstm_preds = scaler.inverse_transform(lstm_preds_scaled).flatten()
            actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            
            progress_bar.progress(100)
            status_text.text("Training complete!")
            
            # ===== CALCULATE METRICS =====
            def calc_metrics(actual, predicted):
                mse = mean_squared_error(actual, predicted)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(actual, predicted)
                mape = mean_absolute_percentage_error(actual, predicted) * 100
                r2 = r2_score(actual, predicted)
                
                # Directional accuracy
                actual_dir = np.diff(actual) > 0
                pred_dir = np.diff(predicted) > 0
                dir_acc = np.mean(actual_dir == pred_dir) * 100
                
                return {
                    'RMSE': rmse, 'MAE': mae, 'MAPE': mape,
                    'R2': r2, 'DA': dir_acc
                }
            
            metrics_arima = calc_metrics(actual_prices, np.array(arima_preds))
            metrics_lstm = calc_metrics(actual_prices, lstm_preds)
            
            # Save to session state
            st.session_state['trained'] = True
            st.session_state['arima_preds'] = arima_preds
            st.session_state['lstm_preds'] = lstm_preds
            st.session_state['actual_prices'] = actual_prices
            st.session_state['metrics_arima'] = metrics_arima
            st.session_state['metrics_lstm'] = metrics_lstm
            st.session_state['history_lstm'] = history_lstm
            st.session_state['test_actual'] = test_actual
            
        st.success("Both models trained successfully!")
        st.balloons()
    
    # Display results if trained
    if st.session_state.get('trained', False):
        st.markdown("---")
        st.subheader("Training Results")
        
        metrics_arima = st.session_state['metrics_arima']
        metrics_lstm = st.session_state['metrics_lstm']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ARIMA Model")
            st.metric("RMSE", f"${metrics_arima['RMSE']:.2f}")
            st.metric("MAE", f"${metrics_arima['MAE']:.2f}")
            st.metric("MAPE", f"{metrics_arima['MAPE']:.3f}%")
            st.metric("R² Score", f"{metrics_arima['R2']:.4f}")
            st.metric("Directional Accuracy", f"{metrics_arima['DA']:.2f}%")
        
        with col2:
            st.markdown("### LSTM Model")
            st.metric("RMSE", f"${metrics_lstm['RMSE']:.2f}")
            st.metric("MAE", f"${metrics_lstm['MAE']:.2f}")
            st.metric("MAPE", f"{metrics_lstm['MAPE']:.3f}%")
            st.metric("R² Score", f"{metrics_lstm['R2']:.4f}")
            st.metric("Directional Accuracy", f"{metrics_lstm['DA']:.2f}%")
        
        # LSTM Training History
        st.markdown("---")
        st.subheader("LSTM Training History")
        
        history_lstm = st.session_state['history_lstm']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        
        axes[0].plot(history_lstm.history['loss'], label='Training Loss', color='#1f77b4', linewidth=2)
        axes[0].plot(history_lstm.history['val_loss'], label='Validation Loss', color='#ff7f0e', linewidth=2)
        axes[0].set_title('Loss Curve', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history_lstm.history['loss'], label='Training Loss', color='#1f77b4', linewidth=2)
        axes[1].plot(history_lstm.history['val_loss'], label='Validation Loss', color='#ff7f0e', linewidth=2)
        axes[1].set_title('Loss Curve (Log Scale)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss (MSE)')
        axes[1].set_yscale('log')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        with st.expander("Penjelasan Visualisasi: Training History"):
            st.markdown("""
            **Mengapa jenis plot ini?**
            - Loss curve adalah standar dalam deep learning untuk monitoring training progress
            - Mendeteksi **overfitting** (training loss << validation loss)
            - Mendeteksi **underfitting** (both losses tinggi)
            - Log scale memudahkan melihat perubahan ketika loss sudah kecil
            
            **Aesthetics & Mapping:**
            - **X-axis:** Epoch number (Discrete/Sequential)
            - **Y-axis:** Loss value MSE (Continuous)
            - **Color (Hue):** Dataset type (Discrete: Training vs Validation)
            - **Line width:** 2px untuk clarity
            
            **Pemilihan Warna (Discrete):**
            - **Biru (#1f77b4) = Training Loss**
            - **Oranye (#ff7f0e) = Validation Loss**
            - Alasan: Standar industri, high contrast untuk membedakan
            
            **Interpretasi:**
            - **Good training:** Kedua lines turun bersamaan, gap kecil
            - **Overfitting:** Training turun terus, validation naik/flat
            - **Underfitting:** Kedua lines tinggi dan flat
            - **Perfect:** Validation loss mendekati training loss di akhir
            - Dari grafik terlihat model konvergen dengan baik tanpa overfitting signifikan
            """)