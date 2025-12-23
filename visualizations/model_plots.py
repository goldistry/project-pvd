import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_training_history(history, model_name):
    """
    Plot training history untuk deep learning models
    """
    # Import color palette
    COLOR_PALETTE = {
        'primary': '#2E86AB',
        'accent': '#F18F01',
        'secondary': '#A23B72'
    }
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f'{model_name} Training History', f'{model_name} Training History (Log Scale)']
    )
    
    # Normal scale
    fig.add_trace(
        go.Scatter(x=list(range(len(history.history['loss']))), 
                  y=history.history['loss'], name='Training Loss',
                  line=dict(color=COLOR_PALETTE['primary'], width=2.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=list(range(len(history.history['val_loss']))), 
                  y=history.history['val_loss'], name='Validation Loss',
                  line=dict(color=COLOR_PALETTE['accent'], width=2.5)),
        row=1, col=1
    )
    
    # Log scale
    fig.add_trace(
        go.Scatter(x=list(range(len(history.history['loss']))), 
                  y=history.history['loss'], name='Training Loss',
                  line=dict(color=COLOR_PALETTE['primary'], width=2.5), showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=list(range(len(history.history['val_loss']))), 
                  y=history.history['val_loss'], name='Validation Loss',
                  line=dict(color=COLOR_PALETTE['accent'], width=2.5), showlegend=False),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss (MSE)", row=1, col=1)
    fig.update_yaxes(title_text="Loss (MSE)", type="log", row=1, col=2)
    
    fig.update_layout(
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def plot_predictions_comparison(actual, predictions_dict, zoom=None):
    """
    Plot perbandingan prediksi semua model
    
    Parameters:
    -----------
    actual : array
        Nilai aktual
    predictions_dict : dict
        Dict dengan key = model name, value = predictions
    zoom : int, optional
        Jumlah hari untuk zoom view
    """
    # Professional color palette for models
    colors = {
        'ARIMA': '#A23B72',    # Deep Purple
        'LSTM': '#2E86AB',     # Professional Blue  
        'GRU': '#06A77D'       # Teal Green
    }
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Model Predictions Comparison (Full Test Period)', 
                       f'Zoom View: First {zoom or min(100, len(actual))} Days']
    )
    
    # Full predictions
    fig.add_trace(
        go.Scatter(x=list(range(len(actual))), y=actual, name='Actual',
                  line=dict(color='#34495E', width=3)),
        row=1, col=1
    )
    
    for model_name, preds in predictions_dict.items():
        fig.add_trace(
            go.Scatter(x=list(range(len(preds))), y=preds, 
                      name=f'{model_name} Prediction',
                      line=dict(color=colors.get(model_name, '#E74C3C'), width=2.5)),
            row=1, col=1
        )
    
    # Zoom view
    if zoom is None:
        zoom = min(100, len(actual))
    
    fig.add_trace(
        go.Scatter(x=list(range(zoom)), y=actual[:zoom], name='Actual',
                  line=dict(color='#34495E', width=3), showlegend=False,
                  mode='lines+markers', marker=dict(size=4)),
        row=2, col=1
    )
    
    for model_name, preds in predictions_dict.items():
        fig.add_trace(
            go.Scatter(x=list(range(zoom)), y=preds[:zoom], 
                      name=f'{model_name} Prediction',
                      line=dict(color=colors.get(model_name, '#E74C3C'), width=2.5),
                      showlegend=False, mode='lines+markers', marker=dict(size=4)),
            row=2, col=1
        )
    
    fig.update_xaxes(title_text="Day Index", row=1, col=1)
    fig.update_xaxes(title_text="Day Index", row=2, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=2, col=1)
    
    fig.update_layout(
        height=700,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def plot_residuals(actual, predictions_dict):
    """
    Plot residual analysis untuk semua model
    """
    fig = go.Figure()
    
    colors = {'ARIMA': '#9467bd', 'LSTM': '#1f77b4', 'GRU': '#2ca02c'}
    
    for model_name, preds in predictions_dict.items():
        residuals = np.array(actual).flatten() - np.array(preds).flatten()
        fig.add_trace(
            go.Scatter(x=list(range(len(residuals))), y=residuals, 
                      name=f'{model_name} Residuals',
                      line=dict(color=colors.get(model_name, '#d62728'), width=1))
        )
    
    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.5)
    
    fig.update_layout(
        title='Residual Analysis (Actual - Predicted)',
        xaxis_title='Day Index',
        yaxis_title='Residual (USD)',
        height=400
    )
    
    return fig

def plot_metrics_comparison(metrics_dict):
    """
    Plot comparison bar charts untuk semua metrik
    """
    models = list(metrics_dict.keys())
    colors = {
        'ARIMA': '#A23B72',    # Deep Purple
        'LSTM': '#2E86AB',     # Professional Blue  
        'GRU': '#06A77D'       # Teal Green
    }
    color_list = [colors.get(m, '#E74C3C') for m in models]
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=['MAPE - Lower is Better', 'RMSE - Lower is Better', 'MAE - Lower is Better',
                       'R2 Score - Higher is Better', 'Directional Accuracy - Higher is Better', 'Summary']
    )
    
    # MAPE
    mape_values = [metrics_dict[m]['MAPE'] for m in models]
    fig.add_trace(
        go.Bar(x=models, y=mape_values, name='MAPE', 
               marker_color=color_list, showlegend=False,
               text=[f'{v:.3f}%' for v in mape_values], textposition='outside'),
        row=1, col=1
    )
    
    # RMSE
    rmse_values = [metrics_dict[m]['RMSE'] for m in models]
    fig.add_trace(
        go.Bar(x=models, y=rmse_values, name='RMSE', 
               marker_color=color_list, showlegend=False,
               text=[f'{v:.2f}' for v in rmse_values], textposition='outside'),
        row=1, col=2
    )
    
    # MAE
    mae_values = [metrics_dict[m]['MAE'] for m in models]
    fig.add_trace(
        go.Bar(x=models, y=mae_values, name='MAE', 
               marker_color=color_list, showlegend=False,
               text=[f'{v:.2f}' for v in mae_values], textposition='outside'),
        row=1, col=3
    )
    
    # R2 Score
    r2_values = [metrics_dict[m]['R2'] for m in models]
    fig.add_trace(
        go.Bar(x=models, y=r2_values, name='R2', 
               marker_color=color_list, showlegend=False,
               text=[f'{v:.4f}' for v in r2_values], textposition='outside'),
        row=2, col=1
    )
    
    # Directional Accuracy
    da_values = [metrics_dict[m]['DA'] for m in models]
    fig.add_trace(
        go.Bar(x=models, y=da_values, name='DA', 
               marker_color=color_list, showlegend=False,
               text=[f'{v:.2f}%' for v in da_values], textposition='outside'),
        row=2, col=2
    )
    
    # Summary table
    table_data = []
    metrics_names = ['MAPE (%)', 'RMSE', 'MAE', 'R2', 'Dir.Acc']
    
    for metric, key in zip(metrics_names, ['MAPE', 'RMSE', 'MAE', 'R2', 'DA']):
        row = [metric]
        values = [metrics_dict[m][key] for m in models]
        
        if key in ['MAPE', 'RMSE', 'MAE']:
            best_idx = values.index(min(values))
        else:
            best_idx = values.index(max(values))
        
        for i, (m, v) in enumerate(zip(models, values)):
            if key in ['MAPE', 'DA']:
                formatted = f"{v:.2f}%"
            elif key in ['RMSE', 'MAE']:
                formatted = f"{v:.2f}"
            else:
                formatted = f"{v:.4f}"
            
            if i == best_idx:
                formatted += " (Best)"
            row.append(formatted)
        
        table_data.append(row)
    
    fig.add_trace(
        go.Table(
            header=dict(values=['Metric'] + models, fill_color='#BDC3C7'),
            cells=dict(values=list(zip(*table_data)), fill_color='white')
        ),
        row=2, col=3
    )
    
    fig.update_layout(
        height=700,
        title_text="Comprehensive Model Evaluation Metrics",
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig