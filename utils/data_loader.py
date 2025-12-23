import pandas as pd
import streamlit as st

@st.cache_data
def load_data(file_path='dataset/indexProcessed.csv'):
    """
    Load and prepare the stock index dataset
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe with Date column parsed
    """
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        st.error(f"File dataset tidak ditemukan di {file_path}")
        return pd.DataFrame()

def get_index_statistics(df):
    """
    Calculate statistics for all indices in the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
        
    Returns:
    --------
    pd.DataFrame
        Statistics for each index
    """
    stats_list = []
    
    for idx in df['Index'].unique():
        subset = df[df['Index'] == idx].sort_values('Date')
        
        # Calculate duration in years
        duration = (subset['Date'].max() - subset['Date'].min()).days / 365.25
        
        # Calculate volatility (standard deviation of daily returns)
        daily_return = subset['Close'].pct_change()
        volatility = daily_return.std() * 100
        
        stats_list.append({
            'Index': idx,
            'Start Year': subset['Date'].min().year,
            'Duration (Years)': round(duration, 1),
            'Avg Volume': subset['Volume'].mean(),
            'Volatility (%)': round(volatility, 3),
            'Total Data': len(subset)
        })
    
    comp_df = pd.DataFrame(stats_list).sort_values('Total Data', ascending=False)
    return comp_df

def filter_index_data(df, index_name):
    """
    Filter data for a specific index
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    index_name : str
        Name of the index to filter
        
    Returns:
    --------
    pd.DataFrame
        Filtered and sorted data with Date as index
    """
    data = df[df['Index'] == index_name].sort_values('Date').reset_index(drop=True)
    data.set_index('Date', inplace=True)
    return data