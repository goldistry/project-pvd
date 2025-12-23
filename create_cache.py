#!/usr/bin/env python3
"""
Script untuk pre-training dan caching semua model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.model_cache import train_and_cache_models

def main():
    """Pre-train models untuk semua indeks utama"""
    
    print("="*60)
    print("PRE-TRAINING MODELS UNTUK CACHE")
    print("="*60)
    print("Script ini akan train semua model sekali dan save ke cache")
    print()
    
    # Indeks yang akan di-cache (hanya NYA untuk testing)
    indices = ['GDAXI']
    
    for i, index in enumerate(indices, 1):
        print(f"\n[{i}/{len(indices)}] Training models untuk {index}...")
        print("-" * 40)
        
        try:
            # Train dengan parameter optimal
            results = train_and_cache_models(
                index_name=index,
                train_ratio=0.8,
                window_size=60,
                epochs=50
            )
            
            print(f" {index} berhasil di-cache!")
            print(f"   - ARIMA RMSE: {results['metrics_arima']['RMSE']:.2f}")
            print(f"   - LSTM RMSE: {results['metrics_lstm']['RMSE']:.2f}")
            print(f"   - GRU RMSE: {results['metrics_gru']['RMSE']:.2f}")
            
        except Exception as e:
            print(f" Error training {index}: {e}")
            continue
    
    print("\n" + "="*60)
    print("CACHE CREATION COMPLETE!")
    print("="*60)
    print("Steps setelah punya cache:")
    print("1. Buka Streamlit: streamlit run app.py")
    print("2. Pilih index apapun")
    print("3. Centang 'Use cached models (faster)'")
    print("4. Klik 'Train All Models' (akan dapat hasil langsung)")
    print()
    print("Cache location: ./model_cache/")
    print("="*60)

if __name__ == "__main__":
    main()