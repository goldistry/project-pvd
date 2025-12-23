import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def plot_future_forecast(data, future_df, model_name="ARIMA", history_days=100):
    """
    Plot future price forecast with confidence interval
    
    Parameters:
    -----------
    data : pd.DataFrame
        Historical data
    future_df : pd.DataFrame
        Forecast data with columns: Forecast, Lower_Bound, Upper_Bound
    model_name : str
        Name of the model
    history_days : int
        Number of historical days to show
        
    Returns:
    --------
    plotly.graph_objects.Figure, float
    """
    # Get last N days of historical data
    history_zoom = data['Close'].iloc[-history_days:]
    
    # Calculate trend
    last_close = data['Close'].iloc[-1]
    final_forecast = future_df['Forecast'].iloc[-1]
    percentage_change = ((final_forecast - last_close) / last_close) * 100
    
    # Determine trend color
    if percentage_change > 0:
        trend_color = '#00CC66'
        trend_label = f"BULLISH (+{percentage_change:.2f}%)"
        fill_color = 'rgba(0, 204, 102, 0.15)'
    else:
        trend_color = '#FF3333'
        trend_label = f"BEARISH ({percentage_change:.2f}%)"
        fill_color = 'rgba(255, 51, 51, 0.15)'
    
    fig = go.Figure()
    
    # Plot historical data
    fig.add_trace(
        go.Scatter(x=history_zoom.index, y=history_zoom, name='Historical Data',
                  line=dict(color='#1f77b4', width=2))
    )
    
    # Plot connection line
    fig.add_trace(
        go.Scatter(x=[history_zoom.index[-1], future_df.index[0]],
                  y=[history_zoom.iloc[-1], future_df['Forecast'].iloc[0]],
                  line=dict(color=trend_color, dash='dash', width=2),
                  showlegend=False)
    )
    
    # Plot forecast
    fig.add_trace(
        go.Scatter(x=future_df.index, y=future_df['Forecast'], 
                  name=f'{model_name} Forecast ({trend_label})',
                  line=dict(color=trend_color, width=3))
    )
    
    # Plot confidence interval
    fig.add_trace(
        go.Scatter(x=future_df.index, y=future_df['Upper_Bound'],
                  fill=None, mode='lines', line_color='rgba(0,0,0,0)',
                  showlegend=False)
    )
    fig.add_trace(
        go.Scatter(x=future_df.index, y=future_df['Lower_Bound'],
                  fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
                  name='Confidence Interval (95%)', fillcolor=fill_color)
    )
    
    # Mark final forecast point
    fig.add_trace(
        go.Scatter(x=[future_df.index[-1]], y=[final_forecast],
                  mode='markers+text', marker=dict(color=trend_color, size=10),
                  text=[f'Target: ${final_forecast:,.2f}'],
                  textposition='middle right', showlegend=False)
    )
    
    # Add vertical line
    fig.add_vline(x=history_zoom.index[-1], line_dash="dot", 
                 line_color="grey", opacity=0.5)
    
    fig.update_layout(
        title=f'60-Day Price Forecast: {trend_label}',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        height=600,
        hovermode='x unified',
        margin=dict(r=100)
    )
    
    return fig, percentage_change

def plot_forecast_breakdown(future_df):
    """
    Plot detailed forecast breakdown showing daily predictions
    
    Parameters:
    -----------
    future_df : pd.DataFrame
        Forecast data
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Daily Forecast Values', 'Daily Price Change Forecast'],
        vertical_spacing=0.1
    )
    
    # Daily forecast values
    fig.add_trace(
        go.Scatter(x=future_df.index, y=future_df['Forecast'], 
                  name='Forecast', line=dict(color='#1f77b4', width=2),
                  mode='lines+markers', marker=dict(size=4)),
        row=1, col=1
    )
    
    # Confidence interval
    fig.add_trace(
        go.Scatter(x=future_df.index, y=future_df['Upper_Bound'],
                  fill=None, mode='lines', line_color='rgba(0,0,0,0)',
                  showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=future_df.index, y=future_df['Lower_Bound'],
                  fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
                  name='Confidence Interval', fillcolor='rgba(31, 119, 180, 0.2)'),
        row=1, col=1
    )
    
    # Daily change
    daily_change = future_df['Forecast'].diff()
    colors = ['green' if x > 0 else 'red' for x in daily_change]
    
    fig.add_trace(
        go.Bar(x=future_df.index, y=daily_change, 
               marker_color=colors, opacity=0.6, name='Daily Change'),
        row=2, col=1
    )
    
    fig.add_hline(y=0, line_dash="solid", line_color="black", row=2, col=1)
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Price Change (USD)", row=2, col=1)
    
    fig.update_layout(height=700)
    
    return fig

def plot_forecast_statistics(future_df, current_price):
    """
    Plot forecast statistics and distribution
    
    Parameters:
    -----------
    future_df : pd.DataFrame
        Forecast data
    current_price : float
        Current price for comparison
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Distribution of Forecast Prices', 'Weekly Average Forecast']
    )
    
    # Price range distribution
    fig.add_trace(
        go.Histogram(x=future_df['Forecast'], nbinsx=20, 
                    marker_color='#1f77b4', opacity=0.7, name='Distribution'),
        row=1, col=1
    )
    
    fig.add_vline(x=current_price, line_dash="dash", line_color="red",
                 annotation_text=f'Current: ${current_price:.2f}', row=1, col=1)
    fig.add_vline(x=future_df['Forecast'].mean(), line_dash="dash", line_color="green",
                 annotation_text=f'Avg: ${future_df["Forecast"].mean():.2f}', row=1, col=1)
    
    # Weekly summary
    future_df_copy = future_df.copy()
    future_df_copy['Week'] = range(1, len(future_df_copy) + 1)
    future_df_copy['Week'] = (future_df_copy['Week'] - 1) // 7 + 1
    
    weekly_avg = future_df_copy.groupby('Week')['Forecast'].mean()
    weekly_std = future_df_copy.groupby('Week')['Forecast'].std()
    
    fig.add_trace(
        go.Scatter(x=weekly_avg.index, y=weekly_avg.values,
                  mode='lines+markers', marker=dict(size=8),
                  line=dict(color='#1f77b4', width=2), name='Weekly Avg'),
        row=1, col=2
    )
    
    # Add error bars for std
    fig.add_trace(
        go.Scatter(x=weekly_avg.index, y=weekly_avg + weekly_std,
                  fill=None, mode='lines', line_color='rgba(0,0,0,0)',
                  showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=weekly_avg.index, y=weekly_avg - weekly_std,
                  fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
                  name='±1 Std Dev', fillcolor='rgba(31, 119, 180, 0.3)'),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_xaxes(title_text="Week", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Average Price (USD)", row=1, col=2)
    
    fig.update_layout(height=500)
    
    return fig