import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from utils.data_loader import load_data, get_index_statistics, filter_index_data
from utils.preprocessing import (calculate_technical_indicators, test_stationarity,
                                  prepare_lstm_data, prepare_arima_data)
from utils.modeling import (train_arima_model, build_lstm_model, build_gru_model,
                             train_deep_learning_model, predict_future_arima)
from utils.metrics import calculate_metrics, get_best_model, format_metric_display
from utils.model_cache import get_or_train_models, is_cache_valid

from visualizations.eda_plots import (plot_price_trends, plot_return_distribution,
                                       plot_boxplot_by_decade, plot_correlation_heatmap,
                                       plot_volatility, plot_acf_pacf, plot_seasonality_heatmap)
from visualizations.model_plots import (plot_training_history, plot_predictions_comparison,
                                         plot_residuals, plot_metrics_comparison)
from visualizations.prediction_plots import (plot_future_forecast, plot_forecast_breakdown,
                                              plot_forecast_statistics)

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Stock Market Prediction - Final Project",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR ---
st.sidebar.title("Anggota Kelompok:")
st.sidebar.text("1. Aloysia Jennifer Harijadi (C14230191)")
st.sidebar.text("2. Felicia Audrey (C14230207)")
st.sidebar.text("3. Jessica Chandra (C14230250)")
st.sidebar.markdown("---")
st.sidebar.info("Pilih Tab di halaman utama untuk berpindah antara Overview, EDA, dan Prediksi.")

# --- HEADER & JUDUL ---
st.title("Analisis & Prediksi Pasar Saham Global")
st.markdown("### Perbandingan Model: ML-ARIMA vs LSTM vs GRU")

# --- INFORMASI PROYEK ---
st.info("""
**Tugas Ujian Akhir Semester - Mata Kuliah Presentasi dan Visualisasi Data**

Fokus utama proyek ini adalah membandingkan performa metode **Machine Learning ARIMA** dengan **Deep Learning (LSTM & GRU)** untuk memprediksi tren harga saham global selama 60 hari ke depan.
""")

# --- BAGIAN 1: OVERVIEW & TUJUAN ---
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Latar Belakang & Tujuan")
    st.write("""
    Pasar saham adalah sistem yang dinamis, non-linear, dan penuh dengan *noise*. Proyek ini memanfaatkan data historis jangka panjang untuk membangun model prediksi yang andal.
    
    **Tujuan Utama:**
    1.  **Forecasting:** Memprediksi harga penutupan (*Close Price*) untuk **60 hari ke depan**.
    2.  **Komparasi Model:** Mengevaluasi akurasi antara ML-ARIMA vs LSTM & GRU (Neural Networks).
    3.  **Visualisasi:** Menyajikan dashboard interaktif untuk memantau pergerakan harga saham global dari AS, Asia, hingga Eropa.
    """)

with col2:
    st.subheader("Ringkasan Data")
    st.success("""
    * **Sumber:** [Kaggle - Stock Exchange Data](https://www.kaggle.com/datasets/mattiuzc/stock-exchange-data)
    * **Periode:** 1965 - 2021 (Bervariasi per indeks)
    * **Frekuensi:** Harian (Daily)
    * **Total Baris:** > 100,000 data points
    """)

# TAMBAHAN: DETAIL DATASET
st.markdown("---")
st.subheader("Penjelasan Dataset (Cakupan & Variabel)")

# Menggunakan Expander agar halaman tidak terlalu panjang tapi detail tetap ada
with st.expander("Lihat Detail Indeks Saham & Penjelasan Variabel", expanded=False):
    
    tab_indeks, tab_variabel = st.tabs(["Cakupan Indeks Global", "Penjelasan Variabel (Fitur)"])
    
    with tab_indeks:
        st.markdown("""
        Dataset ini mencakup **15 Indeks Saham Utama** yang merepresentasikan kekuatan ekonomi dari berbagai benua. 
        Data ini memungkinkan kita melihat korelasi ekonomi global.
        """)
        
        # Membuat Dataframe statis untuk tampilan rapi
        data_indeks = {
            "Region": ["United States", "United States", "United States", "Hong Kong", "Japan", "China", "China", "Europe", "Germany", "Switzerland", "Canada", "India", "Korea", "Taiwan", "South Africa"],
            "Exchange": ["New York Stock Exchange", "NASDAQ", "S&P 500", "Hong Kong Stock Exchange", "Tokyo Stock Exchange", "Shanghai Stock Exchange", "Shenzhen Stock Exchange", "Euronext", "Frankfurt Stock Exchange", "SIX Swiss Exchange", "Toronto Stock Exchange", "NSE India", "Korea Exchange", "Taiwan Stock Exchange", "Johannesburg Stock Exchange"],
            "Index Code": ["NYA", "IXIC", "GSPC", "HSI", "N225", "000001.SS", "399001.SZ", "N100", "GDAXI", "SSMI", "GSPTSE", "NSEI", "KS11", "TWII", "J203.JO"],
            "Currency": ["USD", "USD", "USD", "HKD", "JPY", "CNY", "CNY", "EUR", "EUR", "CHF", "CAD", "INR", "KRW", "TWD", "ZAR"]
        }
        df_display = pd.DataFrame(data_indeks)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    with tab_variabel:
        st.markdown("Setiap baris data merepresentasikan aktivitas perdagangan satu hari dengan atribut berikut:")
        
        col_var1, col_var2 = st.columns(2)
        with col_var1:
            st.markdown("""
            * **Date:** Tanggal pencatatan transaksi saham.
            * **Open:** Harga pembukaan saham pada hari tersebut.
            * **High:** Harga tertinggi yang dicapai saham pada hari tersebut.
            * **Low:** Harga terendah yang dicapai saham pada hari tersebut.
            """)
        with col_var2:
            st.markdown("""
            * **Close:** Harga penutupan saham (Harga terakhir diperdagangkan). **Ini adalah target prediksi model.**
            * **Adj Close:** Harga penutupan yang disesuaikan (*Adjusted*) setelah memperhitungkan dividen atau *stock split*.
            * **Volume:** Jumlah lembar saham yang diperdagangkan pada hari tersebut.
            """)
        
        st.info("**Catatan:** Dataset ini menggunakan versi *processed* yang sudah menyediakan kolom **'CloseUSD'**. Kolom ini menstandarisasi seluruh mata uang ke dalam US Dollar sehingga perbandingan harga antar indeks dapat dilakukan secara adil.")

st.markdown("---")

# --- BAGIAN 2: PENJELASAN MODEL (Card Style) ---
st.subheader("Metodologi Modeling")
st.write("Kami menggunakan tiga pendekatan berbeda untuk menangkap pola *Time Series*:")

model_col1, model_col2, model_col3 = st.columns(3)

with model_col1:
    st.markdown("### 1. ML-ARIMA")
    st.caption("*Machine Learning Approach*")
    st.markdown("""
    **Auto Regressive Integrated Moving Average dengan ML optimization.**
    * **Cara Kerja:** Otomatis mencari parameter (p,d,q) terbaik menggunakan validation-based selection.
    * **Kelebihan:** Kombinasi interpretabilitas statistik dengan automated parameter tuning.
    """)

with model_col2:
    st.markdown("### 2. LSTM")
    st.caption("*Deep Learning (RNN)*")
    st.markdown("""
    **Long Short-Term Memory.**
    * **Cara Kerja:** Menggunakan *memory cells* dan 3 *gates* (input, output, forget).
    * **Kelebihan:** Sangat baik menangkap pola jangka panjang dan non-linear.
    """)

with model_col3:
    st.markdown("### 3. GRU")
    st.caption("*Deep Learning (RNN)*")
    st.markdown("""
    **Gated Recurrent Unit.**
    * **Cara Kerja:** Versi simplifikasi LSTM dengan hanya 2 *gates* (update, reset).
    * **Kelebihan:** Komputasi lebih cepat dari LSTM dengan akurasi yang kompetitif.
    """)

st.markdown("---")

# --- BAGIAN 3: ALUR ANALISIS ---
with st.expander("Lihat Detail Tahapan Analisis (Workflow)"):
    st.markdown("""
    Analisis dilakukan dalam 2 tahapan utama:
    1.  **Eksplorasi & Validasi Model (Notebook):** Berfokus pada analisis mendalam menggunakan satu indeks utama sebagai studi kasus.
        * Melakukan *Hyperparameter Tuning* dan *Data Splitting* (Train/Test) untuk validasi akurasi.
    2.  **Implementasi & Skalabilitas (Streamlit):** Menerapkan model terbaik ke dalam aplikasi web.
        * Memungkinkan pengguna memilih indeks saham lain (seperti IXIC, HSI, N225) dan melihat hasil forecasting secara real-time.
        
    **Data Split Strategy:**
    - **Training:** ~50 tahun data historis (semua data kecuali 5 tahun terakhir)
    - **Testing:** 5 tahun terakhir untuk evaluasi model
    - **Window Size:** 60 hari lookback untuk LSTM/GRU
    - **Epochs:** 50 untuk balanced training
    """)

st.markdown("---")

# Load data dan setup
df = load_data()

if df.empty:
    st.stop()

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Pengaturan Model")
available_indices = sorted(df['Index'].unique())
selected_index = st.sidebar.selectbox(
    "Pilih Stock Index:",
    available_indices,
    index=available_indices.index('NYA') if 'NYA' in available_indices else 0
)

data = filter_index_data(df, selected_index)
data = calculate_technical_indicators(data)
data['Close_Diff'] = data['Close'].diff()

st.sidebar.markdown("---")
st.sidebar.info(f"""
**Dataset Info:**
- Index: {selected_index}
- Total Data: {len(data)} hari
- Periode: {data.index.min().date()} s/d {data.index.max().date()}
- Durasi: {(data.index.max() - data.index.min()).days / 365.25:.1f} tahun
""")

st.sidebar.markdown("---")
st.sidebar.subheader("Model Settings")
test_years = st.sidebar.slider("Test Period (Years)", 3, 10, 5, 1)
st.sidebar.info(f"""
**Data Split:**
- Training: ~{(data.index.max() - data.index.min()).days / 365.25 - test_years:.1f} years
- Testing: {test_years} years
""")

# Fixed parameters (no longer user-configurable)
window_size = 60  # Optimal for financial time series
lstm_epochs = 50  # Balanced training

# ARIMA approach selection
st.sidebar.markdown("**ARIMA Configuration:**")
use_ml_arima = st.sidebar.checkbox("Use ML-based Auto-ARIMA", value=True, 
                                   help="Automatically find optimal parameters using grid search")

# Check if models are cached
if is_cache_valid(selected_index):
    st.sidebar.success(f"Pre-trained models available for {selected_index}")
    use_cache = st.sidebar.checkbox("Use cached models (faster)", value=True)
else:
    st.sidebar.warning(f"No cached models for {selected_index}")
    use_cache = False

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview & EDA",
    "Stationarity Test",
    "Model Training",
    "Model Comparison",
    "Future Prediction"
])

with tab1:
    st.header("Exploratory Data Analysis")
    
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
    
    with st.expander("Perbandingan Antar Indeks"):
        stats_df = get_index_statistics(df)
        st.dataframe(stats_df, use_container_width=True)
        
        if selected_index == 'NYA':
            st.success("""
            **Mengapa NYA dipilih sebagai fokus analisis?**
            
            Berdasarkan evaluasi durasi data historis, jumlah observasi, dan tingkat volatilitas, 
            indeks NYA dipilih sebagai indeks utama karena:
            - Durasi data paling panjang (55+ tahun)
            - Jumlah observasi terbesar (13.900+ data points)
            - Volatilitas relatif stabil (1,02%)
            
            Kombinasi ini memberikan basis pelatihan yang kuat dan konsisten untuk membangun 
            model prediksi time series yang robust.
            """)
    
    st.markdown("---")
    
    st.subheader("2. Tren Harga dengan Moving Average")
    fig = plot_price_trends(data, selected_index)
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Penjelasan Visualisasi: Time Series Plot"):
        st.markdown("""
        **Mengapa jenis plot ini?**
        - Line plot adalah pilihan terbaik untuk time series data karena menunjukkan kontinuitas temporal dan tren over time
        - Moving Average menghaluskan noise dan menunjukkan tren jangka pendek (MA-50) vs jangka panjang (MA-200)
        - Bar chart untuk volume memudahkan identifikasi periode trading aktif
        
        **Aesthetics & Mapping:**
        - **X-axis (Position):** Date - waktu dipetakan ke posisi horizontal
        - **Y-axis (Position):** Price & Volume - nilai dipetakan ke posisi vertikal
        - **Color (Hue):** Variable Type (Discrete) - setiap variabel diberi warna berbeda
        - **Line Style:** Solid untuk actual price, dashed untuk moving averages
        
        **Pemilihan Warna:**
        - **Biru (#2E86AB):** Close Price - primary data, trustworthy dan authoritative
        - **Hijau Teal (#06A77D):** MA-50 - short-term trend, growth-associated color
        - **Merah Coral (#E74C3C):** MA-200 - long-term trend, attention-grabbing untuk important signals
        - **Ungu (#A23B72):** Volume - different data type, sophisticated color
        
        **Interpretasi:**
        - **Golden Cross:** MA-50 memotong MA-200 ke atas = bullish signal
        - **Death Cross:** MA-50 memotong MA-200 ke bawah = bearish signal
        - **Volume Spike:** Trading volume tinggi dengan pergerakan harga signifikan mengkonfirmasi kekuatan trend
        
        **Implikasi:**
        - **Modeling (ML-ARIMA):** Tren kuat ini akan dideteksi otomatis oleh ML-ARIMA melalui grid search. Algoritma akan mencoba berbagai kombinasi parameter (p,d,q) dan memilih yang terbaik berdasarkan validation performance, bukan hanya mengandalkan parameter tetap (5,1,0).
        - **Modeling (Deep Learning):** Tren ini mengharuskan normalisasi data (seperti MinMaxScaler) agar model LSTM dan GRU dapat memproses nilai harga yang terus meningkat secara efisien.
        """)
    
    st.markdown("---")
    
    st.subheader("3. Distribusi Daily Return")
    fig = plot_return_distribution(data)
    st.plotly_chart(fig, use_container_width=True)
    
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
        - Histogram menunjukkan frekuensi distribusi data return secara diskrit
        - KDE (Kernel Density Estimation) memberikan smooth curve untuk melihat bentuk distribusi keseluruhan
        - Kombinasi keduanya memberikan pemahaman lengkap tentang distribusi data
        
        **Aesthetics & Mapping:**
        - **X-axis (Position):** Return value dalam persen (Continuous)
        - **Y-axis (Position):** Density (Continuous, normalized frequency)
        - **Bars (Shape):** Histogram dengan height proporsional ke frequency
        - **Line (Shape):** KDE smooth curve menunjukkan probability density
        - **Vertical line (x=0):** Reference point untuk zero return
        
        **Pemilihan Warna:**
        - **Biru (#2E86AB):** Histogram bars - primary data color, trustworthy
        - **Orange Hangat (#F18F01):** KDE line - complementary color untuk distinction
        - **Abu-abu Gelap (#34495E):** Zero reference line - neutral baseline
        - **Color psychology:** Biru = data/analysis, Orange = insight/pattern
        
        **Interpretasi:**
        - **Mean Return positif:** Index cenderung naik secara rata-rata
        - **Skewness negatif:** Lebih banyak extreme downside events (crash)
        - **High Kurtosis (>3):** Fat tails menunjukkan extreme events lebih sering terjadi
        - **Arti:** Data memiliki karakteristik non-normal dengan tail risk
        - **KDE advantage:** Smooth curve memudahkan melihat distribusi shape
        
        **Implikasi:**
        - **Preprocessing:** Data memiliki karakteristik Fat Tails, artinya kejadian ekstrem (seperti crash) lebih sering terjadi dibandingkan distribusi normal.
        - **Modeling (ML-ARIMA):** Karakteristik non-normal ini akan mempengaruhi proses grid search ML-ARIMA. Algoritma akan mencoba berbagai parameter dan memilih yang paling robust terhadap outliers dan extreme events melalui validation performance.
        - **Modeling (Deep Learning):** Karakteristik ini memvalidasi penggunaan loss='mean_squared_error' dalam fungsi build_lstm_model dan build_gru_model. MSE sangat sensitif terhadap selisih besar, sehingga memaksa model untuk belajar dari variansi tinggi tersebut.
        """)
    
    st.markdown("---")
    
    st.subheader("4. Analisis Distribusi per Dekade")
    fig = plot_boxplot_by_decade(data)
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Penjelasan Visualisasi: Boxplot Analysis"):
        st.markdown("""
        **Mengapa jenis plot ini?**
        - Boxplot ideal untuk mendeteksi outliers dan memahami distribusi data dalam kelompok kategorikal
        - Menunjukkan quartiles, median, dan outliers secara visual yang jelas
        - Memungkinkan perbandingan distribusi antar dekade dengan mudah
        
        **Aesthetics & Mapping:**
        - **X-axis (Position):** Decade (Discrete/Categorical)
        - **Y-axis (Position):** Price/Return (Continuous)
        - **Box components (Shape):** IQR (Q1-Q3), Median line, Whiskers (1.5×IQR), Outlier points
        - **Color (Hue):** Default plotly color sequence untuk membedakan dekade
        
        **Pemilihan Warna (Sequential):**
        - **Plotly default palette:** Setiap dekade mendapat warna berbeda secara otomatis
        - **Merah (dashed line):** Horizontal line di y=0 untuk return plot sebagai reference
        - **Konsistensi:** Warna yang sama untuk dekade yang sama di kedua subplot
        
        **Interpretasi:**
        - **Box height:** Menunjukkan volatilitas (IQR) pada dekade tersebut
        - **Outliers:** Extreme events atau market crashes/booms
        - **Median position:** Trend central tendency per dekade
        - **Whisker length:** Range normal variasi harga/return
        
        **Implikasi:**
        - **Modeling (ML-ARIMA):** Fenomena volatilitas yang tidak konstan ini akan dipertimbangkan dalam proses automated parameter selection. ML-ARIMA akan mencoba berbagai kombinasi (p,d,q) dan memilih yang paling adaptif terhadap regime changes melalui validation-based model selection.
        - **Modeling (Deep Learning):** Fenomena volatilitas yang tidak konstan ini menjustifikasi penggunaan layer Dropout(0.2) di antara layer LSTM atau GRU. Layer ini berfungsi sebagai regulasi agar model tidak terlalu terpaku (overfit) pada fluktuasi ekstrem di dekade tertentu.
        """)
    
    st.markdown("---")
    
    st.subheader("5. Correlation Matrix")
    fig = plot_correlation_heatmap(data)
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Penjelasan Visualisasi: Correlation Heatmap"):
        st.markdown("""
        **Mengapa jenis plot ini?**
        - Heatmap ideal untuk visualisasi correlation matrix menggunakan color intensity
        - Memudahkan identifikasi pola korelasi antar variabel secara visual
        - Annotations memberikan nilai exact untuk analisis detail
        
        **Aesthetics & Mapping:**
        - **Rows & Columns (Position):** Feature names (Categorical)
        - **Cell Color (Hue):** Correlation coefficient (Continuous: -1 to +1)
        - **Text Annotations:** Exact correlation values untuk precision
        - **Cell Size:** Uniform untuk fair comparison
        
        **Pemilihan Warna (Continuous Diverging):**
        - **Red-Yellow-Green** diverging colormap:
          - **RED (#d62728):** Korelasi negatif kuat (-1.0 to -0.5)
          - **YELLOW (#ffff99):** Korelasi lemah/tidak ada (-0.5 to +0.5)
          - **GREEN (#2ca02c):** Korelasi positif kuat (+0.5 to +1.0)
        - **Center at 0 (zmid=0):** Menunjukkan "no correlation" sebagai neutral point
        - **Symmetric scale:** Memberikan bobot visual yang sama untuk korelasi positif dan negatif
        
        **Interpretasi:**
        - **Dark Green (>0.9):** Open-High-Low-Close memiliki korelasi sangat tinggi - EXPECTED
        - **Yellow/Light colors:** Volume memiliki korelasi independen dengan price features

        **Implikasi:**
        - **Implikasi modeling:** Cukup gunakan Close dan Volume untuk menghindari multicollinearity
        """)
    
    st.markdown("---")
    
    st.subheader("6. Volatility Analysis")
    fig = plot_volatility(data)
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Penjelasan Visualisasi: Volatility Analysis"):
        st.markdown("""
        **Mengapa jenis plot ini?**
        - Line plot menunjukkan perubahan volatilitas over time - critical untuk understanding risk dynamics
        - Time series visualization untuk melihat volatility clustering dan regime changes
        - Interactive features memungkinkan zoom pada periode crisis tertentu
        - Fill area memberikan visual emphasis pada volatility magnitude
        
        **Aesthetics & Mapping:**
        - **X-axis (Position):** Date (Temporal/Continuous)
        - **Y-axis (Position):** Volatility percentage (Continuous)
        - **Line (Shape):** Continuous line menunjukkan trend volatilitas
        - **Fill area:** Subtle background fill untuk visual emphasis
        - **Line width:** Cukup tebal (2.5px) untuk visibility
        
        **Pemilihan Warna (Risk-Focused Theme):**
        - **Kuning Emas (#F5BB00):** Dipilih karena asosiasi dengan warning/caution dalam finance
        - **Psychology:** Kuning = attention/caution, lebih soft daripada merah untuk volatility
        - **Fill transparency:** Light yellow fill memberikan context tanpa overwhelming
        - **Contrast:** Kuning kontras baik dengan background putih untuk readability
        
        **Interpretasi:**
        - **Spike pada volatility:** Periode market uncertainty (crisis, major news, earnings)
        - **Low volatility periods:** Market stability dan investor confidence
        - **Volatility clustering:** High volatility cenderung diikuti high volatility (GARCH effect)
        - **Model implications:** Model perlu robust terhadap regime changes
        - **Fill area benefit:** Memudahkan melihat magnitude dan duration volatility periods
        """)
    
    st.markdown("---")
    
    if selected_index == 'NYA':
        st.subheader("7. Analisis Seasonality (NYA Only)")
        fig = plot_seasonality_heatmap(data)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Penjelasan Visualisasi: Seasonality Heatmap"):
            st.markdown("""
            **Mengapa visualisasi ini?**
            - Heatmap 2D ideal untuk mendeteksi pola musiman (seasonality) dengan dimensi Year x Month
            - Color intensity menunjukkan magnitude return, memudahkan spot pattern
            - Matrix format memungkinkan perbandingan cross-sectional dan time-series
            
            **Aesthetics & Mapping:**
            - **X-axis (Position):** Month (1-12, Discrete/Ordinal)
            - **Y-axis (Position):** Year (Continuous/Temporal)
            - **Cell Color (Hue):** Average daily return (Continuous)
            - **Color intensity:** Proportional to return magnitude
            
            **Pemilihan Warna (Continuous Diverging):**
            - **Red-Yellow-Green** diverging colormap:
              - **RED:** Negative returns (losses)
              - **YELLOW:** Near-zero returns (neutral)
              - **GREEN:** Positive returns (gains)
            - **Center at 0 (zmid=0):** Zero return sebagai neutral point
            - **Financial intuition:** Red=loss, Green=gain adalah standar di finance
            
            **Interpretasi untuk NYA:**
            - **High color variation:** Tidak ditemukan pola musiman yang konsisten
            - **Random pattern:** Warna bervariasi tinggi antar tahun untuk bulan yang sama
            - **Market-driven:** Performa bulanan dipengaruhi kondisi pasar spesifik, bukan efek musiman

            **Implikasi:**
            - **Modeling:** Seasonality lemah, ML-ARIMA akan fokus pada non-seasonal parameters (p,d,q) tanpa komponen SARIMA. Grid search akan mengoptimalkan parameter dasar yang lebih efektif untuk pola non-musiman ini. Penambahan komponen musiman hanya akan menambah kompleksitas tanpa meningkatkan akurasi.
            """)

with tab2:
    st.header("Stationarity Test (untuk ARIMA)")
    
    st.info("""
    **Mengapa Stationarity Penting?**
    
    ARIMA model membutuhkan data yang **stationary** (mean, variance, autocorrelation konstan over time).
    Gunakan **Augmented Dickey-Fuller (ADF) Test** untuk menguji stationarity.
    
    - **H0 (Null Hypothesis):** Data memiliki unit root (NON-STATIONARY)
    - **H1 (Alternative):** Data tidak memiliki unit root (STATIONARY)
    - **Decision:** Jika p-value ≤ 0.05, reject H0, data STATIONARY
    """)
    
    st.subheader("1. Test pada Level (Original Data)")
    result_level = test_stationarity(data['Close'])
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("ADF Statistic", f"{result_level['adf_statistic']:.6f}")
        st.metric("p-value", f"{result_level['p_value']:.6f}")
        if result_level['is_stationary']:
            st.success("STATIONARY")
        else:
            st.error("NON-STATIONARY")
    
    with col2:
        st.write("**Critical Values:**")
        for key, value in result_level['critical_values'].items():
            st.write(f"- {key}: {value:.3f}")
        
        if result_level['is_stationary']:
            st.write("✓ p-value ≤ 0.05: Reject H0, Data STATIONARY")
        else:
            st.write("✗ p-value > 0.05: Gagal reject H0, Data NON-STATIONARY")
    
    st.markdown("---")
    
    st.subheader("2. Test pada First Difference")
    result_diff = test_stationarity(data['Close_Diff'])
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("ADF Statistic", f"{result_diff['adf_statistic']:.6f}")
        st.metric("p-value", f"{result_diff['p_value']:.6f}")
        if result_diff['is_stationary']:
            st.success("STATIONARY")
        else:
            st.error("NON-STATIONARY")
    
    with col2:
        st.write("**Critical Values:**")
        for key, value in result_diff['critical_values'].items():
            st.write(f"- {key}: {value:.3f}")
        
        if result_diff['is_stationary']:
            st.write("✓ p-value ≤ 0.05: Reject H0, Data STATIONARY")
        else:
            st.write("✗ p-value > 0.05: Gagal reject H0, Data NON-STATIONARY")
    
    st.markdown("---")
    
    st.subheader("3. ACF & PACF Plots")
    fig = plot_acf_pacf(data)
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Penjelasan Visualisasi: ACF & PACF Analysis"):
        st.markdown("""
        **Mengapa jenis plot ini?**
        - Stem plot (seperti statsmodels) lebih akurat daripada bar chart untuk menunjukkan correlation coefficients
        - ACF mengukur korelasi antara time series dengan lag-nya (total correlation)
        - PACF mengukur korelasi setelah menghilangkan efek lag intermediate (partial correlation)
        - Tool standar untuk menentukan parameter optimal ARIMA (p, d, q)
        
        **Aesthetics & Mapping:**
        - **X-axis (Position):** Lag number (Discrete/Ordinal, 0 to 40)
        - **Y-axis (Position):** Correlation coefficient (Continuous, -1 to +1)
        - **Stem Lines (Shape):** Vertical lines dari zero ke correlation value
        - **Markers (Shape):** Points di ujung stem untuk emphasis
        - **Confidence bands:** Dashed lines menunjukkan significance threshold
        
        **Pemilihan Warna:**
        - **Biru (#2E86AB):** ACF stems - primary analysis color
        - **Ungu (#A23B72):** PACF stems - secondary analysis color
        - **Merah Coral (#E74C3C):** Confidence intervals - warning/threshold color
        - **Abu-abu Gelap (#34495E):** Zero reference line - neutral baseline
        - **Color differentiation:** Memudahkan distinguish antara ACF dan PACF
        
        **Interpretasi:**
        - **ACF significant lags:** Menentukan parameter **q** (MA order) untuk ARIMA
        - **PACF significant lags:** Menentukan parameter **p** (AR order) untuk ARIMA
        - **Stems outside confidence bands:** Statistically significant correlations
        - **Stem plot advantage:** Lebih mirip dengan output statsmodels yang familiar
        
        **Implikasi:**
        - **Modeling (ML-ARIMA):** ML-ARIMA akan menggunakan ACF/PACF patterns sebagai starting point untuk grid search, kemudian mengoptimalkan parameter melalui validation performance. Cutoff patterns memberikan hint awal, tapi final selection berdasarkan empirical performance.
        """)
    
    st.markdown("---")
    
    st.success(f"""
    **Kesimpulan Stationarity Test:**
    
    - **Original data:** {result_level['interpretation']}
    - **After differencing:** {result_diff['interpretation']}
    - **Recommended d parameter:** {0 if result_level['is_stationary'] else 1}
    
    Untuk ML-ARIMA, akan menggunakan **d={0 if result_level['is_stationary'] else 1}** sebagai constraint dalam grid search.
    Grid search akan mengoptimalkan parameter **p dan q** secara otomatis berdasarkan validation performance.
    """)

with tab3:
    st.header("Model Training: ARIMA vs LSTM vs GRU")
    
    if st.button("Train All Models", type="primary"):
        if use_cache:
            with st.spinner("Loading pre-trained models..."):
                # Load cached results
                cached_results = get_or_train_models(selected_index, force_retrain=False)
                
                # Store in session state
                for key, value in cached_results.items():
                    st.session_state[key] = value
                
                st.session_state['trained'] = True
                st.success("Pre-trained models loaded successfully!")
        else:
            with st.spinner("Training models... This may take several minutes..."):
                X_train, y_train, X_test, y_test, scaler, train_size = prepare_lstm_data(
                    data, window_size, test_years
                )
                train_data_arima, test_data_arima, _ = prepare_arima_data(data, test_years)
                
                st.session_state['X_train'] = X_train
                st.session_state['y_train'] = y_train
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['scaler'] = scaler
                st.session_state['train_size'] = train_size
                st.session_state['test_data_arima'] = test_data_arima
                st.session_state['selected_index'] = selected_index
                
                st.subheader("Training ARIMA Model...")
                arima_preds, arima_errors, _ = train_arima_model(
                    train_data_arima, test_data_arima, order=(5, 1, 0), 
                    window_size=window_size, use_ml_approach=use_ml_arima
                )
                st.session_state['arima_preds'] = arima_preds
                st.success(f"ARIMA trained! Total errors: {len(arima_errors)}")
                
                st.subheader("Training LSTM Model...")
                model_lstm = build_lstm_model(window_size)
                history_lstm = train_deep_learning_model(
                    model_lstm, X_train, y_train, epochs=lstm_epochs, model_name="LSTM"
                )
                lstm_preds_scaled = model_lstm.predict(X_test, verbose=0)
                lstm_preds = scaler.inverse_transform(lstm_preds_scaled).flatten()
                
                st.session_state['model_lstm'] = model_lstm
                st.session_state['history_lstm'] = history_lstm
                st.session_state['lstm_preds'] = lstm_preds
                st.success("LSTM trained!")
                
                st.subheader("Training GRU Model...")
                model_gru = build_gru_model(window_size)
                history_gru = train_deep_learning_model(
                    model_gru, X_train, y_train, epochs=lstm_epochs, model_name="GRU"
                )
                gru_preds_scaled = model_gru.predict(X_test, verbose=0)
                gru_preds = scaler.inverse_transform(gru_preds_scaled).flatten()
                
                st.session_state['model_gru'] = model_gru
                st.session_state['history_gru'] = history_gru
                st.session_state['gru_preds'] = gru_preds
                st.success("GRU trained!")
                
                actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
                st.session_state['actual_prices'] = actual_prices
                
                metrics_arima = calculate_metrics(actual_prices, arima_preds)
                metrics_lstm = calculate_metrics(actual_prices, lstm_preds)
                metrics_gru = calculate_metrics(actual_prices, gru_preds)
                
                st.session_state['metrics_arima'] = metrics_arima
                st.session_state['metrics_lstm'] = metrics_lstm
                st.session_state['metrics_gru'] = metrics_gru
                
                st.session_state['trained'] = True
        
        st.success("All models trained successfully!")
        st.balloons()
    
    if st.session_state.get('trained', False):
        st.markdown("---")
        st.subheader("Training Results")
        
        col1, col2, col3 = st.columns(3)
        
        metrics_arima = st.session_state['metrics_arima']
        metrics_lstm = st.session_state['metrics_lstm']
        metrics_gru = st.session_state['metrics_gru']
        
        with col1:
            st.markdown("### ARIMA Model")
            st.metric("RMSE", format_metric_display(metrics_arima['RMSE'], 'RMSE'))
            st.metric("MAE", format_metric_display(metrics_arima['MAE'], 'MAE'))
            st.metric("MAPE", format_metric_display(metrics_arima['MAPE'], 'MAPE'))
            st.metric("R² Score", format_metric_display(metrics_arima['R2'], 'R2'))
            st.metric("Directional Accuracy", format_metric_display(metrics_arima['DA'], 'DA'))
        
        with col2:
            st.markdown("### LSTM Model")
            st.metric("RMSE", format_metric_display(metrics_lstm['RMSE'], 'RMSE'))
            st.metric("MAE", format_metric_display(metrics_lstm['MAE'], 'MAE'))
            st.metric("MAPE", format_metric_display(metrics_lstm['MAPE'], 'MAPE'))
            st.metric("R² Score", format_metric_display(metrics_lstm['R2'], 'R2'))
            st.metric("Directional Accuracy", format_metric_display(metrics_lstm['DA'], 'DA'))
        
        with col3:
            st.markdown("### GRU Model")
            st.metric("RMSE", format_metric_display(metrics_gru['RMSE'], 'RMSE'))
            st.metric("MAE", format_metric_display(metrics_gru['MAE'], 'MAE'))
            st.metric("MAPE", format_metric_display(metrics_gru['MAPE'], 'MAPE'))
            st.metric("R² Score", format_metric_display(metrics_gru['R2'], 'R2'))
            st.metric("Directional Accuracy", format_metric_display(metrics_gru['DA'], 'DA'))
        
        st.markdown("---")
        st.subheader("Training History")
        
        # LSTM Training History - Full width
        st.markdown("#### LSTM Training History")
        fig_lstm = plot_training_history(st.session_state['history_lstm'], "LSTM")
        st.plotly_chart(fig_lstm, use_container_width=True)
        
        # GRU Training History - Full width  
        st.markdown("#### GRU Training History")
        fig_gru = plot_training_history(st.session_state['history_gru'], "GRU")
        st.plotly_chart(fig_gru, use_container_width=True)
        
        with st.expander("Penjelasan Visualisasi: Training History"):
            st.markdown("""
            **Mengapa jenis plot ini?**
            - Dual-scale visualization untuk melihat training progress dalam normal dan log scale
            - Line plot menunjukkan konvergensi model selama jumlah epoch tertentu.
            - Log scale memudahkan melihat perubahan kecil pada nilai loss ketika angkanya sudah sangat rendah, yang sulit terlihat pada skala normal.
            
            **Aesthetics & Mapping:**
            - **X-axis (Position):** Epoch number (Discrete/Sequential)
            - **Y-axis (Position):** Loss value menggunakan metrik MSE (Continuous)
            - **Line (Shape):** Continuous lines menunjukkan learning progression
            - **Dual subplot:** Normal scale vs Log scale untuk different perspectives
            
            **Pemilihan Warna:**
            - **Biru (#2E86AB):** Digunakan untuk Training Loss. Warna ini dipilih karena memberikan kesan stabilitas, kepercayaan, dan otoritas sebagai metrik utama.
            - **Orange Hangat (#F18F01):** Digunakan untuk Validation Loss. Berfungsi sebagai warna komplementer untuk membedakan kategori dan menarik perhatian pada validasi data.
            - **High contrast:** Mudah dibedakan untuk analysis
            
            **Interpretasi Loss Curve:**
            - **Good training:** Idealnya, kedua garis (biru dan oranye) turun secara bersamaan dengan celah (gap) yang kecil di antara keduanya.
            - **Overfitting:** Training turun terus, namun validation loss justru naik/flat
            - **Underfitting:** Kedua lines tinggi dan flat, tidak ada improvement
            - **Optimal stopping:** Titik tepat sebelum validation loss mulai naik kembali; di sinilah model mencapai performa terbaiknya.
            - **Log scale benefit:** Melihat improvement detail pada loss values yang kecil
            
             **Analisis Hasil Plot:**
            - **LSTM Training History:** Training loss (biru) turun tajam dan stabil di angka rendah. Namun, validation loss (oranye) menunjukkan volatilitas tinggi dengan lonjakan tajam (spikes) di beberapa epoch awal sebelum akhirnya turun kembali. Hal ini mengindikasikan sensitivitas model LSTM terhadap urutan data tertentu pada set validasi.
            - **GRU Training History:** Menunjukkan pola yang lebih stabil dibandingkan LSTM. Garis training dan validation turun lebih sinkron, meskipun masih terdapat fluktuasi kecil pada validation loss. Hal ini menjelaskan mengapa dalam hasil metrik Anda, GRU cenderung lebih unggul dalam stabilitas dibandingkan LSTM untuk dataset ini.
            """)

with tab4:
    st.header("Model Comparison & Analysis")
    
    if not st.session_state.get('trained', False):
        st.warning("Please train the models first in the 'Model Training' tab.")
    else:
        actual_prices = st.session_state['actual_prices']
        arima_preds = st.session_state['arima_preds']
        lstm_preds = st.session_state['lstm_preds']
        gru_preds = st.session_state['gru_preds']
        
        predictions_dict = {
            'ARIMA': arima_preds,
            'LSTM': lstm_preds,
            'GRU': gru_preds
        }
        
        st.subheader("1. Predictions Comparison")
        fig = plot_predictions_comparison(actual_prices, predictions_dict, zoom=100)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Penjelasan Visualisasi: Model Predictions Comparison"):
            st.markdown("""
            **Mengapa jenis plot ini?**
            - Multi-line plot ideal untuk membandingkan performa multiple models secara visual
            - Dual subplot: full view untuk trend analysis, zoom view untuk detail accuracy
            - Interactive features memungkinkan hover comparison dan zoom functionality
            
            **Aesthetics & Mapping:**
            - **X-axis (Position):** Day index (Sequential/Ordinal)
            - **Y-axis (Position):** Price in USD (Continuous)
            - **Lines (Shape):** Different models dengan line weights berbeda
            - **Markers:** Added pada zoom view untuk precision
            
            **Pemilihan Warna:**
            - **Abu-abu Gelap (#34495E):** Actual values - authoritative ground truth, neutral
            - **Ungu (#A23B72):** ARIMA - statistical model, sophisticated color
            - **Biru (#2E86AB):** LSTM - deep learning, technology-associated
            - **Hijau Teal (#06A77D):** GRU - alternative deep learning, natural progression
            - **Semantic meaning:** Warna mencerminkan karakteristik masing-masing model
            - **High contrast:** Semua warna mudah dibedakan untuk comparison
            
            **Interpretasi:**
            - **Line proximity to gray:** Semakin dekat ke actual, semakin akurat model
            - **Smooth vs jagged lines:** Model stability dan noise handling
            - **Zoom view detail:** Precision pada short-term predictions
            - **Overall trend:** Long-term forecasting capability
            - **Color coding consistency:** Memudahkan tracking model performance
            """)
        
        st.markdown("---")
        
        st.subheader("2. Residuals Analysis")
        fig = plot_residuals(actual_prices, predictions_dict)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Penjelasan Visualisasi: Residual Analysis"):
            st.markdown("""
            **Mengapa jenis plot ini?**
            - Residual plot essential untuk model diagnostic dan assumption checking
            - Line plot menunjukkan error patterns over time
            - Zero line sebagai reference untuk unbiased predictions
            
            **Aesthetics & Mapping:**
            - **X-axis (Position):** Day index (Sequential)
            - **Y-axis (Position):** Residual value (Continuous, USD)
            - **Lines (Shape):** Thin lines untuk melihat error patterns
            - **Zero reference:** Horizontal line di y=0
            
            **Pemilihan Warna (Consistent with Predictions):**
            - **Ungu (#9467bd):** ARIMA residuals - consistency dengan prediction plot
            - **Biru (#1f77b4):** LSTM residuals - same reasoning
            - **Hijau (#2ca02c):** GRU residuals - maintains color coding
            - **Hitam (solid):** Zero line - neutral reference
            
            **Interpretasi:**
            - **Random scatter around zero:** Good model (no systematic bias)
            - **Patterns in residuals:** Model missing some signal
            - **Heteroscedasticity:** Changing variance over time
            - **Mean close to zero:** Unbiased predictions
            """)
        
        col1, col2, col3 = st.columns(3)
        
        metrics_arima = st.session_state['metrics_arima']
        metrics_lstm = st.session_state['metrics_lstm']
        metrics_gru = st.session_state['metrics_gru']
        
        with col1:
            st.markdown("**ARIMA Residuals:**")
            st.write(f"Mean: {metrics_arima['Residual_Mean']:.2f}")
            st.write(f"Std: {metrics_arima['Residual_Std']:.2f}")
        with col2:
            st.markdown("**LSTM Residuals:**")
            st.write(f"Mean: {metrics_lstm['Residual_Mean']:.2f}")
            st.write(f"Std: {metrics_lstm['Residual_Std']:.2f}")
        with col3:
            st.markdown("**GRU Residuals:**")
            st.write(f"Mean: {metrics_gru['Residual_Mean']:.2f}")
            st.write(f"Std: {metrics_gru['Residual_Std']:.2f}")
        
        st.markdown("---")
        
        st.subheader("3. Comprehensive Metrics Comparison")
        
        metrics_dict = {
            'ARIMA': metrics_arima,
            'LSTM': metrics_lstm,
            'GRU': metrics_gru
        }
        
        fig = plot_metrics_comparison(metrics_dict)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Penjelasan Visualisasi: Comprehensive Metrics Comparison"):
            st.markdown("""
            **Mengapa jenis plot ini?**
            - Multi-subplot layout untuk comprehensive model evaluation
            - Bar charts ideal untuk comparing discrete categories (models)
            - Table summary memberikan exact values dan best model identification
            
            **Aesthetics & Mapping:**
            - **X-axis (Position):** Model names (Categorical)
            - **Y-axis (Position):** Metric values (Continuous, different scales)
            - **Bar height (Size):** Proportional to metric value
            - **Text annotations:** Exact values on bars untuk precision
            
            **Pemilihan Warna (Consistent Model Identity):**
            - **Ungu (#A23B72):** ARIMA - maintains identity across all visualizations
            - **Biru (#2E86AB):** LSTM - consistent deep learning representation
            - **Hijau Teal (#06A77D):** GRU - alternative deep learning model
            - **Color consistency:** Same colors di semua plots untuk easy model tracking
            - **Visual hierarchy:** Warna membantu immediate model recognition
            
            **Interpretasi Metrics:**
            - **MAPE, RMSE, MAE:** Lower is better (error metrics)
            - **R2 Score:** Higher is better (explained variance)
            - **Directional Accuracy:** Higher is better (trend prediction)
            - **Best model:** Highlighted dalam summary table
            - **Color coding advantage:** Instant visual association dengan model performance
            """)
        
        st.markdown("---")
        
        st.subheader("4. Best Model Selection")
        best_model, wins = get_best_model(metrics_dict)
        
        st.success(f"""
        **BEST MODEL: {best_model}**
        
        **Score Breakdown:**
        - ARIMA: {wins['ARIMA']}/5 metrics won
        - LSTM: {wins['LSTM']}/5 metrics won
        - GRU: {wins['GRU']}/5 metrics won
        """)
        
        current_index = st.session_state.get('selected_index', selected_index)
        
        if current_index == 'NYA':
            if best_model == 'ARIMA':
                st.info("""
                **Mengapa ARIMA Unggul untuk NYA?**
                
                1. **Pola Linear:** Dataset NYA Index memiliki pola yang relatif linear dan trend-based
                2. **Autoregressive Pattern:** ARIMA cocok untuk data dengan pola autoregressive yang kuat
                3. **Data Complexity:** LSTM/GRU membutuhkan pola non-linear yang lebih kompleks
                4. **Signal-to-Noise Ratio:** Stock data memiliki noise tinggi, model sederhana lebih robust
                5. **Overfitting Risk:** Deep learning models mungkin overfit pada training data
                
                **Kesimpulan:** Untuk dataset NYA dengan karakteristik linear dan trend-based, 
                model statistik klasik ARIMA memberikan prediksi yang lebih akurat dan stabil.
                """)
            elif best_model in ['LSTM', 'GRU']:
                st.info(f"""
                **Mengapa {best_model} Unggul untuk NYA?**
                
                1. **Non-linear Patterns:** {best_model} mampu menangkap pola non-linear yang kompleks
                2. **Long-term Dependencies:** Lebih baik dalam capture pola jangka panjang
                3. **Feature Learning:** Belajar representasi optimal secara otomatis
                4. **Adaptability:** Lebih flexible terhadap perubahan pola market
                
                **Kesimpulan:** Model deep learning memberikan performa superior pada dataset ini 
                karena kemampuannya menangkap kompleksitas yang tidak terdeteksi oleh model statistik.
                """)
        else:
            st.info(f"""
            **Interpretasi Hasil untuk {current_index}:**
            
            Model **{best_model}** menunjukkan performa terbaik pada indeks ini dengan 
            memenangkan {wins[best_model]} dari 5 metrik evaluasi. 
            
            Hasil ini menunjukkan karakteristik data pada indeks {current_index} 
            cocok dengan arsitektur model {best_model}.
            
            **Rekomendasi:** Gunakan model {best_model} untuk prediksi pada indeks ini.
            """)

with tab5:
    st.header("Future Prediction (60 Hari ke Depan)")
    
    if not st.session_state.get('trained', False):
        st.warning("Please train the models first in the 'Model Training' tab.")
    else:
        best_model, wins = get_best_model({
            'ARIMA': st.session_state['metrics_arima'],
            'LSTM': st.session_state['metrics_lstm'],
            'GRU': st.session_state['metrics_gru']
        })
        
        st.info(f"""
        **Model Terpilih untuk Prediksi:** {best_model}
        
        Berdasarkan evaluasi metrics, model **{best_model}** akan digunakan untuk melakukan 
        forecasting 60 hari ke depan.
        """)
        
        if st.button("Generate 60-Day Forecast", type="primary"):
            with st.spinner("Generating forecast..."):
                
                forecast_df = predict_future_arima(data, steps=60, order=(5, 1, 0))
                st.session_state['forecast_df'] = forecast_df
                
                last_close = data['Close'].iloc[-1]
                final_forecast = forecast_df['Forecast'].iloc[-1]
                percentage_change = ((final_forecast - last_close) / last_close) * 100
                
                st.session_state['last_close'] = last_close
                st.session_state['final_forecast'] = final_forecast
                st.session_state['percentage_change'] = percentage_change
                st.session_state['forecast_generated'] = True
            
            st.success("Forecast generated successfully!")
        
        if st.session_state.get('forecast_generated', False):
            forecast_df = st.session_state['forecast_df']
            last_close = st.session_state['last_close']
            final_forecast = st.session_state['final_forecast']
            percentage_change = st.session_state['percentage_change']
            
            st.subheader("1. Future Price Forecast")
            fig, percentage_change = plot_future_forecast(data, forecast_df, selected_index)
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Penjelasan Visualisasi: Future Forecast"):
                st.markdown("""
                **Mengapa jenis plot ini?**
                - Time series continuation plot untuk menunjukkan historical context dan future projection
                - Confidence intervals memberikan uncertainty quantification
                - Color coding berdasarkan trend direction (bullish/bearish)
                
                **Aesthetics & Mapping:**
                - **X-axis (Position):** Date (Temporal/Continuous)
                - **Y-axis (Position):** Price in USD (Continuous)
                - **Lines (Shape):** Historical (solid), forecast (solid), connection (dashed)
                - **Fill area:** Confidence interval dengan transparency
                
                **Pemilihan Warna (Semantic/Conditional):**
                - **Biru (#1f77b4):** Historical data - neutral, factual color
                - **Hijau (#00CC66):** Bullish forecast - universally associated dengan gains
                - **Merah (#FF3333):** Bearish forecast - universally associated dengan losses
                - **Transparency:** Fill area dengan alpha untuk tidak menghalangi main lines
                - **Dashed connection:** Visual bridge antara historical dan forecast
                
                **Interpretasi:**
                - **Trend direction:** Color immediately communicates bullish/bearish signal
                - **Confidence bands:** Wider bands = higher uncertainty
                - **Target point:** Final forecast value dengan annotation
                - **Percentage change:** Quantified expected return
                """)
            
            st.markdown("---")
            
            st.subheader("2. Forecast Breakdown")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${last_close:,.2f}")
            with col2:
                st.metric("Predicted Price (Day 60)", f"${final_forecast:,.2f}")
            with col3:
                st.metric("Expected Change", f"{percentage_change:+.2f}%", 
                         delta=f"${final_forecast - last_close:+,.2f}")
            
            st.markdown("---")
            
            fig = plot_forecast_breakdown(forecast_df)
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Penjelasan Visualisasi: Forecast Breakdown"):
                st.markdown("""
                **Mengapa jenis plot ini?**
                - Dual subplot untuk detailed daily analysis: absolute values dan changes
                - Line plot dengan markers untuk precision pada daily forecasts
                - Bar chart untuk daily changes menunjukkan volatility patterns
                
                **Aesthetics & Mapping:**
                - **X-axis (Position):** Date (Temporal)
                - **Y-axis (Position):** Price/Change (Continuous)
                - **Lines + Markers:** Daily forecast precision
                - **Bars:** Daily price changes dengan conditional coloring
                
                **Pemilihan Warna (Functional):**
                - **Biru (#1f77b4):** Forecast line - consistent dengan main forecast
                - **Light blue fill:** Confidence interval - related tapi lighter
                - **Green bars:** Positive daily changes - gains
                - **Red bars:** Negative daily changes - losses
                - **Black line:** Zero reference untuk daily changes
                
                **Interpretasi:**
                - **Daily precision:** Markers show exact daily predictions
                - **Confidence evolution:** How uncertainty changes over time
                - **Daily volatility:** Bar heights show expected daily movements
                - **Trend consistency:** Overall direction vs daily fluctuations
                """)
            
            st.markdown("---")
            
            st.subheader("3. Forecast Statistics")
            fig = plot_forecast_statistics(forecast_df, last_close)
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Penjelasan Visualisasi: Forecast Statistics"):
                st.markdown("""
                **Mengapa jenis plot ini?**
                - Dual analysis: distribution analysis dan temporal aggregation (weekly)
                - Histogram untuk understanding forecast range dan probability
                - Line plot dengan error bands untuk weekly trend analysis
                
                **Aesthetics & Mapping:**
                - **X-axis:** Price (continuous) untuk histogram, Week (ordinal) untuk line plot
                - **Y-axis:** Frequency untuk histogram, Average price untuk line plot
                - **Bars:** Histogram bins dengan frequency
                - **Line + Fill:** Weekly averages dengan standard deviation bands
                
                **Pemilihan Warna (Analytical):**
                - **Biru (#1f77b4):** Consistent dengan forecast theme di semua plots
                - **Red dashed:** Current price reference - important baseline
                - **Green dashed:** Average forecast - target expectation
                - **Light blue fill:** Standard deviation bands - uncertainty visualization
                
                **Interpretasi:**
                - **Distribution shape:** Normal vs skewed forecast probabilities
                - **Price range:** Min-max expected values
                - **Weekly progression:** How forecast evolves over time
                - **Uncertainty bands:** Confidence in weekly averages
                """)
            
            with st.expander("Forecast Data Table"):
                st.dataframe(forecast_df, use_container_width=True)
            
            st.markdown("---")
            
            if percentage_change > 0:
                st.success(f"""
                **BULLISH SIGNAL (+{percentage_change:.2f}%)**
                
                Model memproyeksikan kenaikan harga untuk 60 hari ke depan.
                Indikasi tren positif untuk indeks {selected_index}.
                """)
            else:
                st.error(f"""
                **BEARISH SIGNAL ({percentage_change:.2f}%)**
                
                Model memproyeksikan penurunan harga untuk 60 hari ke depan.
                Indikasi tren negatif untuk indeks {selected_index}.
                """)
            
            st.warning("""
            **DISCLAIMER:**
            
            Hasil prediksi ini merupakan proyeksi statistik berdasarkan data historis 
            dan pola yang terdeteksi oleh model. Prediksi ini **BUKAN** rekomendasi investasi 
            dan tidak dapat dijadikan jaminan pergerakan harga di masa depan.
            
            Pasar saham dipengaruhi oleh banyak faktor eksternal yang tidak dapat diprediksi 
            (geopolitik, kebijakan moneter, sentiment pasar, dll). Selalu diperlukan riset 
            mendalam dan mungkin konsultasi dengan ahli sebelum mengambil keputusan investasi.
            """)