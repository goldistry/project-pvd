import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

# Color Palette - Consistent Theme
COLOR_PALETTE = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Deep Purple
    'accent': '#F18F01',       # Warm Orange
    'success': '#06A77D',      # Teal Green
    'warning': '#F5BB00',      # Golden Yellow
    'danger': '#E74C3C',       # Coral Red
    'neutral': '#34495E',      # Dark Gray
    'light': '#BDC3C7'        # Light Gray
}

def plot_price_trends(data, index_name):
    """
    Plot price trends with moving averages and volume
    
    Parameters:
    -----------
    data : pd.DataFrame
        Stock data with MA columns
    index_name : str
        Name of the index
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[f'Price Trend: {index_name}', 'Trading Volume'],
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )
    
    # Price with MA
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'], name='Close Price',
                  line=dict(color=COLOR_PALETTE['primary'], width=2.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['MA_50'], name='MA-50',
                  line=dict(color=COLOR_PALETTE['success'], width=2, dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['MA_200'], name='MA-200',
                  line=dict(color=COLOR_PALETTE['danger'], width=2, dash='dash')),
        row=1, col=1
    )
    
    # Volume
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume',
               marker_color=COLOR_PALETTE['secondary'], opacity=0.6),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def plot_return_distribution(data):
    """
    Plot distribution of daily returns with histogram and KDE
    
    Parameters:
    -----------
    data : pd.DataFrame
        Stock data with Daily_Return column
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    returns = data['Daily_Return'].dropna()
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=returns, nbinsx=100, name='Distribution',
            marker_color=COLOR_PALETTE['primary'], opacity=0.7,
            histnorm='probability density'
        )
    )
    
    # Add KDE curve using scipy
    try:
        from scipy import stats
        import numpy as np
        
        # Create KDE
        kde = stats.gaussian_kde(returns)
        x_range = np.linspace(returns.min(), returns.max(), 200)
        kde_values = kde(x_range)
        
        fig.add_trace(
            go.Scatter(
                x=x_range, y=kde_values, 
                mode='lines', name='KDE',
                line=dict(color=COLOR_PALETTE['accent'], width=3)
            )
        )
    except ImportError:
        pass
    
    # Add vertical line at zero
    fig.add_vline(x=0, line_dash="dash", line_color=COLOR_PALETTE['neutral'], opacity=0.8)
    
    fig.update_layout(
        title='Distribution of Daily Returns',
        xaxis_title='Daily Return (%)',
        yaxis_title='Density',
        height=400,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def plot_boxplot_by_decade(data):
    """
    Plot boxplots of price and returns by decade
    
    Parameters:
    -----------
    data : pd.DataFrame
        Stock data with Decade column
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Price Distribution by Decade', 'Daily Return Distribution by Decade']
    )
    
    # Price boxplot
    for decade in sorted(data['Decade'].unique()):
        decade_data = data[data['Decade'] == decade]['Close']
        fig.add_trace(
            go.Box(y=decade_data, name=str(decade), showlegend=False),
            row=1, col=1
        )
    
    # Return boxplot
    for decade in sorted(data['Decade'].unique()):
        decade_data = data[data['Decade'] == decade]['Daily_Return']
        fig.add_trace(
            go.Box(y=decade_data, name=str(decade), showlegend=False),
            row=1, col=2
        )
    
    # Add horizontal line at zero for returns
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5, row=1, col=2)
    
    fig.update_xaxes(title_text="Decade", row=1, col=1)
    fig.update_xaxes(title_text="Decade", row=1, col=2)
    fig.update_yaxes(title_text="Close Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Daily Return (%)", row=1, col=2)
    
    fig.update_layout(height=500)
    
    return fig

def plot_correlation_heatmap(data):
    """
    Plot correlation heatmap of numeric features
    
    Parameters:
    -----------
    data : pd.DataFrame
        Stock data
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Select numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['Decade']]
    
    corr_matrix = data[numeric_cols].corr()
    
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdYlGn',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        )
    )
    
    fig.update_layout(
        title='Correlation Heatmap',
        height=600,
        width=600
    )
    
    return fig

def plot_volatility(data):
    """
    Plot rolling volatility
    
    Parameters:
    -----------
    data : pd.DataFrame
        Stock data with Volatility_30 column
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=data.index, y=data['Volatility_30'],
            name='30-Day Volatility',
            line=dict(color=COLOR_PALETTE['warning'], width=2.5),
            fill='tonexty',
            fillcolor=f"rgba({int(COLOR_PALETTE['warning'][1:3], 16)}, {int(COLOR_PALETTE['warning'][3:5], 16)}, {int(COLOR_PALETTE['warning'][5:7], 16)}, 0.1)"
        )
    )
    
    fig.update_layout(
        title='Rolling Volatility (30-Day)',
        xaxis_title='Date',
        yaxis_title='Volatility (%)',
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def plot_acf_pacf(data):
    """
    Plot ACF and PACF for determining ARIMA parameters
    
    Parameters:
    -----------
    data : pd.DataFrame
        Stock data with Close_Diff column
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    from statsmodels.tsa.stattools import acf, pacf
    
    series = data['Close_Diff'].dropna()
    lags = 40
    
    # Calculate ACF and PACF
    acf_vals, acf_confint = acf(series, nlags=lags, alpha=0.05)
    pacf_vals, pacf_confint = pacf(series, nlags=lags, alpha=0.05)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['ACF - For Determining MA Parameter (q)', 
                       'PACF - For Determining AR Parameter (p)'],
        vertical_spacing=0.15
    )
    
    # ACF plot - using stem plot style like statsmodels
    for i in range(len(acf_vals)):
        fig.add_trace(
            go.Scatter(x=[i, i], y=[0, acf_vals[i]], 
                      mode='lines', line=dict(color='#2E86AB', width=2),
                      showlegend=False),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[i], y=[acf_vals[i]], 
                      mode='markers', marker=dict(color='#2E86AB', size=6),
                      showlegend=False),
            row=1, col=1
        )
    
    # PACF plot - using stem plot style like statsmodels
    for i in range(len(pacf_vals)):
        fig.add_trace(
            go.Scatter(x=[i, i], y=[0, pacf_vals[i]], 
                      mode='lines', line=dict(color='#A23B72', width=2),
                      showlegend=False),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=[i], y=[pacf_vals[i]], 
                      mode='markers', marker=dict(color='#A23B72', size=6),
                      showlegend=False),
            row=2, col=1
        )
    
    # Add confidence intervals
    upper_bound_acf = acf_confint[:, 1]
    lower_bound_acf = acf_confint[:, 0]
    upper_bound_pacf = pacf_confint[:, 1]
    lower_bound_pacf = pacf_confint[:, 0]
    
    fig.add_trace(
        go.Scatter(x=list(range(len(upper_bound_acf))), y=upper_bound_acf,
                  mode='lines', line=dict(color='#E74C3C', dash='dash', width=1),
                  name='95% Confidence', showlegend=True),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=list(range(len(lower_bound_acf))), y=lower_bound_acf,
                  mode='lines', line=dict(color='#E74C3C', dash='dash', width=1),
                  showlegend=False),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=list(range(len(upper_bound_pacf))), y=upper_bound_pacf,
                  mode='lines', line=dict(color='#E74C3C', dash='dash', width=1),
                  showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=list(range(len(lower_bound_pacf))), y=lower_bound_pacf,
                  mode='lines', line=dict(color='#E74C3C', dash='dash', width=1),
                  showlegend=False),
        row=2, col=1
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="#34495E", line_width=1, 
                 opacity=0.8, row=1, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="#34495E", line_width=1, 
                 opacity=0.8, row=2, col=1)
    
    fig.update_xaxes(title_text="Lag", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=2, col=1)
    fig.update_yaxes(title_text="Correlation", row=1, col=1)
    fig.update_yaxes(title_text="Correlation", row=2, col=1)
    
    fig.update_layout(height=700, showlegend=True)
    
    return fig

def plot_seasonality_heatmap(data):
    """
    Plot seasonality heatmap for detecting monthly patterns
    
    Parameters:
    -----------
    data : pd.DataFrame
        Stock data
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    df_heatmap = data.copy()
    df_heatmap['Year'] = df_heatmap.index.year
    df_heatmap['Month'] = df_heatmap.index.month
    df_heatmap['Return'] = df_heatmap['Close'].pct_change()
    
    # Pivot table: Year x Month
    pivot_table = df_heatmap.pivot_table(
        values='Return', 
        index='Year', 
        columns='Month', 
        aggfunc='mean'
    )
    
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_table.values,
            x=[f'Month {i}' for i in pivot_table.columns],
            y=pivot_table.index,
            colorscale='RdYlGn',
            zmid=0,
            hoverongaps=False,
            colorbar=dict(title='Avg Daily Return')
        )
    )
    
    fig.update_layout(
        title='Seasonality Analysis: Average Return by Month',
        xaxis_title='Month (1=January, 12=December)',
        yaxis_title='Year',
        height=600
    )
    
    return fig