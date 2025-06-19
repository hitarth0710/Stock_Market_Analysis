import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flask import jsonify
from matplotlib.ticker import FuncFormatter
import traceback

# Configure matplotlib
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

def generate_basic_plots(data):
    """
    Generate basic stock data visualizations.
    
    Parameters:
        data (dict): Dictionary containing stock data
    
    Returns:
        JSON response with encoded plot images
    """
    try:
        print(f"Generating plots with data keys: {list(data.keys())}")
        
        # Check required keys
        required_keys = ['ticker', 'dates', 'prices', 'volume']
        for key in required_keys:
            if key not in data:
                return jsonify({
                    'status': 'error', 
                    'message': f'Missing required data: {key}'
                })
        
        ticker = data['ticker']
        dates = pd.to_datetime(data['dates'])
        
        # Ensure we have 1D arrays by flattening if necessary
        try:
            prices = np.array(data['prices']).flatten()
            volume = np.array(data['volume']).flatten()
        except Exception as e:
            print(f"Error flattening arrays: {e}")
            # Fallback method
            prices = [float(p) for p in data['prices']]
            volume = [float(v) for v in data['volume']]
        
        # Use alternative field or calculate if volume_ma_20 is missing
        if 'volume_ma_20' in data and data['volume_ma_20']:
            try:
                volume_ma_20 = np.array(data['volume_ma_20']).flatten()
            except:
                volume_ma_20 = [float(v) for v in data['volume_ma_20']]
        else:
            # Calculate manually without using pandas rolling
            window = 20
            volume_ma_20 = []
            for i in range(len(volume)):
                if i < window - 1:
                    # For the first window-1 points, just use whatever data we have
                    volume_ma_20.append(sum(volume[:i+1]) / (i+1))
                else:
                    volume_ma_20.append(sum(volume[i-window+1:i+1]) / window)
            print("Calculated volume_ma_20 as fallback")
            
        # Handle the avg_volume field
        if 'avg_volume' in data and data['avg_volume']:
            avg_volume = float(data['avg_volume'])
        else:
            avg_volume = sum(volume) / len(volume) if len(volume) > 0 else 0
            print("Calculated avg_volume as fallback")
            
        # Handle other optional fields with proper conversion
        try:
            returns = np.array(data.get('returns', [0] * len(prices))).flatten()
            ma_50 = np.array(data.get('ma_50', [0] * len(prices))).flatten()
            ma_200 = np.array(data.get('ma_200', [0] * len(prices))).flatten()
            high = np.array(data.get('high', prices)).flatten()
            low = np.array(data.get('low', prices)).flatten()
            open_prices = np.array(data.get('open', prices)).flatten()
        except:
            # Fallback conversion
            returns = [float(r) if r is not None else 0 for r in data.get('returns', [0] * len(prices))]
            ma_50 = [float(m) if m is not None else 0 for m in data.get('ma_50', [0] * len(prices))]
            ma_200 = [float(m) if m is not None else 0 for m in data.get('ma_200', [0] * len(prices))]
            high = [float(h) if h is not None else p for h, p in zip(data.get('high', prices), prices)]
            low = [float(l) if l is not None else p for l, p in zip(data.get('low', prices), prices)]
            open_prices = [float(o) if o is not None else p for o, p in zip(data.get('open', prices), prices)]
        
        # A container for the plot images
        plots = {}
        
        # Create a DataFrame with all data to simplify plotting
        df_data = {
            'Price': prices,
            'High': high,
            'Low': low,
            'Open': open_prices,
            'Volume': volume,
            'Volume_MA_20': volume_ma_20,
            'Returns': returns,
            'MA_50': ma_50,
            'MA_200': ma_200
        }
        
        # Make sure all arrays are the same length
        min_length = min(len(val) for val in df_data.values())
        for key in df_data:
            df_data[key] = df_data[key][:min_length]
        
        # Also trim dates if needed
        dates = dates[:min_length]
        
        # Create DataFrame
        df = pd.DataFrame(df_data, index=dates)
        
        # Drop rows with missing or invalid values
        df = df.dropna(subset=['Price'])
        
        # Convert any remaining non-numeric values to 0
        numeric_columns = ['Price', 'High', 'Low', 'Open', 'Volume', 'Volume_MA_20', 'Returns', 'MA_50', 'MA_200']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        # Make sure it's sorted by date
        df = df.sort_index()
        
        # Plot 1: Stock Price and Moving Averages
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Price'], label=f'{ticker} Price')
        plt.plot(df.index, df['MA_50'], label='50-Day MA', alpha=0.8)
        plt.plot(df.index, df['MA_200'], label='200-Day MA', alpha=0.8)
        plt.title(f'{ticker} Stock Price and Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plots['price_plot'] = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        # Plot 2: Daily Returns
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(df.index, df['Returns'], color='blue', alpha=0.7)
        plt.title(f'{ticker} Daily Returns')
        plt.xlabel('Date')
        plt.ylabel('Returns (%)')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        sns.histplot(df['Returns'].dropna(), kde=True, color='blue', bins=50)
        plt.title('Distribution of Daily Returns')
        plt.xlabel('Returns (%)')
        plt.tight_layout()
        
        # Save plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plots['returns_plot'] = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        # Plot 3: Trading Volume (enhanced version from notebook)
        plt.figure(figsize=(12, 6))
        ax = plt.subplot(1, 1, 1)
        
        volume_bars = ax.bar(df.index, df['Volume'], color='green', alpha=0.5)
        ax.plot(df.index, df['Volume_MA_20'], color='blue', linewidth=2, label='20-Day MA Volume')
        ax.axhline(y=avg_volume, color='red', linestyle='--', alpha=0.8, 
                   label=f'Avg Volume: {int(avg_volume):,}')
        
        # Identify volume spikes for annotation
        threshold = avg_volume * 2  # 2x average as threshold for spikes
        volume_spikes = df[df['Volume'] > threshold]
        
        # Only annotate recent spikes (last year)
        if not df.empty:
            max_date = df.index.max()
            recent_date = max_date - pd.Timedelta(days=365)
            recent_spikes = volume_spikes[volume_spikes.index >= recent_date]
            
            # Limit to top 5 spikes if there are many
            if len(recent_spikes) > 5:
                recent_spikes = recent_spikes.nlargest(5, 'Volume')
                
            for date, row in recent_spikes.iterrows():
                ax.annotate(f"{row['Volume']/1e6:.1f}M", 
                            xy=(date, row['Volume']),
                            xytext=(0, 10),
                            textcoords="offset points",
                            ha='center', 
                            fontsize=8,
                            arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))
        
        ax.set_title(f'{ticker} Trading Volume')
        ax.set_xlabel('Date')
        ax.set_ylabel('Volume')
        ax.grid(True)
        ax.legend()
        
        # Only use log scale if all volume values are positive and non-zero
        if (df['Volume'] > 0).all() and not df['Volume'].isnull().all():
            try:
                ax.set_yscale('log')
            except ValueError:
                # Fall back to linear scale if log scale fails
                print("Warning: Log scale failed for volume plot, using linear scale")
        
        # Format y-axis labels to show millions
        def millions(x, pos):
            return f'{int(x/1e6)}M' if x >= 1e6 else f'{int(x):,}'
        ax.yaxis.set_major_formatter(FuncFormatter(millions))
        
        plt.tight_layout()
        
        # Save plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plots['volume_plot'] = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        # Plot 4: Candlestick chart for more detailed price analysis
        try:
            import mplfinance as mpf
            
            # Create the OHLC dataframe
            ohlc_data = df[['Open', 'High', 'Low', 'Price']].copy()
            ohlc_data.rename(columns={'Price': 'Close'}, inplace=True)
            
            # Only use last 90 days to avoid memory issues
            recent_data = ohlc_data.iloc[-min(90, len(ohlc_data)):]
            
            # Check if we have valid OHLC data
            if len(recent_data) > 0 and not recent_data.isnull().all().all():
                # Create a custom style
                mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
                s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=False)
                
                # Create a figure and plot the candlestick chart
                fig, ax = plt.subplots(figsize=(12, 6))
                mpf.plot(recent_data, type='candle', style=s, ax=ax,
                       title=f'{ticker} Recent Price Action (Candlestick)',
                       ylabel='Price (USD)')
                
                # Save figure to buffer
                buffer = io.BytesIO()
                fig.savefig(buffer, format='png')
                buffer.seek(0)
                plots['candlestick_plot'] = base64.b64encode(buffer.read()).decode('utf-8')
                plt.close(fig)
            else:
                plots['candlestick_plot'] = None
        except Exception as e:
            print(f"Candlestick plot failed: {str(e)}")
            plots['candlestick_plot'] = None
        
        return jsonify({'status': 'success', 'plots': plots})
        
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error generating plots: {str(e)}")
        print(error_traceback)
        
        return jsonify({
            'status': 'error', 
            'message': str(e), 
            'traceback': error_traceback
        })

def plot_to_base64(fig):
    """
    Convert a matplotlib figure to base64 encoded string.
    
    Parameters:
        fig (Figure): Matplotlib figure
    
    Returns:
        str: Base64 encoded image string
    """
    try:
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    except Exception as e:
        print(f"Error converting plot to base64: {str(e)}")
        # Return a base64 encoded error message image
        plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, f'Error rendering plot: {str(e)}', 
                 horizontalalignment='center', verticalalignment='center')
        plt.axis('off')
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        return base64.b64encode(buffer.read()).decode('utf-8')