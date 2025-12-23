# PANDUAN JALANIN CODE

## Quick Start (Untuk Demo Instant)

### Opsi 1: Gunakan Cache (RECOMMENDED - 30 detik)
```bash
# 1. Jalankan sekali untuk buat cache
python create_cache.py

# 2. Buka Streamlit
streamlit run app.py

# 3. Di sidebar:
#    - Pilih index (NYA/IXIC/DJI/GSPC)
#    - Centang "Use cached models (faster)"
#    - Klik "Train All Models"
#    - INSTANT RESULTS
```

### Opsi 2: Train Fresh (6-10 menit)
```bash
# 1. Buka Streamlit
streamlit run app.py

# 2. Di sidebar:
#    - Pilih index
#    - Jangan centang cache
#    - Klik "Train All Models"
#    - Tunggu 6-10 menit
```

## Model Comparison

| Model | Type | Approach |
|-------|------|----------|
| **ML-ARIMA** | Statistical ML | Automated parameter learning dengan validation |
| **LSTM** | Deep Learning | Recurrent Neural Network untuk sequence modeling |
| **GRU** | Deep Learning | Simplified LSTM dengan fewer parameters |

## Features

**Automated ML-ARIMA**: Parameter optimization via validation  
**Interactive Visualizations**: Plotly charts dengan hover details  
**Comprehensive Metrics**: RMSE, MAE, MAPE, R², Directional Accuracy  
**Future Forecasting**: 60-day predictions dengan confidence intervals  
**Model Caching**: Instant results untuk demo  
**Progress Tracking**: Real-time training progress  

## File Structure
```
project-pvd/
├── app.py                 # Main Streamlit app
├── create_cache.py        # Script untuk buat cache
├── model_cache/          # Cached models (dibuat otomatis)
├── utils/                # Core functions
├── visualizations/       # Plotting functions
└── dataset/             # Stock market data
```

## Troubleshooting

**Error "Module not found":**
```bash
pip install -r requirements.txt
```

**Cache tidak ada:**
```bash
python create_cache.py
```

**Streamlit tidak jalan:**
```bash
pip install streamlit
streamlit run app.py
```

---
**Dibuat untuk Tugas UAS - Presentasi dan Visualisasi Data**