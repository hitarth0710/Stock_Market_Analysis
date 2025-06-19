import io
import base64
import pandas as pd
import numpy as np
import matplotlib
# Set non-interactive backend to avoid Tkinter issues
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from flask import jsonify
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import traceback

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
        img_str = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()  # Explicitly close the buffer
        plt.close(fig)  # Explicitly close the figure
        return img_str
    except Exception as e:
        print(f"Error converting plot to base64: {str(e)}")
        # Return a base64 encoded error message image
        err_fig = plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, f'Error rendering plot: {str(e)}', 
                 horizontalalignment='center', verticalalignment='center')
        plt.axis('off')
        buffer = io.BytesIO()
        err_fig.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()  # Explicitly close the buffer
        plt.close(err_fig)  # Explicitly close the figure
        return img_str

def custom_decompose(series, period=None):
    """
    Custom time series decomposition that works with any amount of data.
    
    Parameters:
        series: Time series data
        period: Seasonality period (will be calculated if None)
    
    Returns:
        dict: Components of the decomposition
    """
    # Determine appropriate window size for trend
    if period is None:
        if len(series) >= 104:  # 2 years of weekly data
            period = 52
        elif len(series) >= 52:  # 1 year of weekly data
            period = 26
        elif len(series) >= 24:  # 6 months of weekly data
            period = 12
        else:
            # For short series use a small fraction of the data length
            period = max(2, len(series) // 4)
    
    # Window size for trend should be larger than period
    trend_window = max(period, min(len(series) // 3, period * 2))
    
    # Make window odd for centered moving average
    if trend_window % 2 == 0:
        trend_window += 1
    
    # Calculate trend component using centered moving average
    trend = series.rolling(window=trend_window, center=True).mean()
    
    # Handle NaN values at edges
    trend = trend.fillna(method='bfill').fillna(method='ffill')
    
    # Detrend the series (either multiplicative or additive)
    if (series > 0).all():
        # Multiplicative model
        seasonal_plus_residual = series / trend
        model_type = "multiplicative"
    else:
        # Additive model
        seasonal_plus_residual = series - trend
        model_type = "additive"
    
    # Try to extract seasonality if we have enough data
    try:
        # Create an array of indices for the time dimension
        time_idx = np.arange(len(series)) % period
        
        # Group by time index and calculate average seasonal component
        grouped = pd.DataFrame({'value': seasonal_plus_residual, 'time_idx': time_idx})
        seasonal_pattern = grouped.groupby('time_idx')['value'].mean()
        
        # Normalize the seasonal component
        if model_type == "multiplicative":
            seasonal_pattern = seasonal_pattern / seasonal_pattern.mean()
        else:
            seasonal_pattern = seasonal_pattern - seasonal_pattern.mean()
        
        # Map the seasonal pattern back to the original time series
        seasonal = pd.Series(index=series.index)
        for i, idx in enumerate(series.index):
            season_idx = i % period
            seasonal.loc[idx] = seasonal_pattern.iloc[season_idx]
        
        # Calculate residual
        if model_type == "multiplicative":
            resid = seasonal_plus_residual / seasonal
        else:
            resid = seasonal_plus_residual - seasonal
            
    except:
        # If seasonal extraction fails, just use the detrended series as residual
        seasonal = pd.Series(1 if model_type == "multiplicative" else 0, index=series.index)
        resid = seasonal_plus_residual
    
    # Return components
    return {
        'observed': series,
        'trend': trend,
        'seasonal': seasonal,
        'resid': resid,
        'model_type': model_type,
        'period': period
    }

def perform_time_series_analysis(data):
    """
    Perform time series decomposition and stationarity tests.
    
    Parameters:
        data (dict): Dictionary containing stock data
    
    Returns:
        JSON response with analysis results and plots
    """
    try:
        print(f"Starting time series analysis with data keys: {list(data.keys())}")
        
        ticker = data['ticker']
        dates = pd.to_datetime(data['dates'])
        
        # Ensure we have 1D arrays by flattening if necessary
        try:
            prices = np.array(data['prices']).flatten()
        except Exception as e:
            print(f"Error flattening price array: {e}")
            # Fallback method
            prices = [float(p) for p in data['prices']]
        
        # Create a DataFrame for easier handling
        df = pd.DataFrame({
            'Date': dates,
            'Price': prices
        })
        df.set_index('Date', inplace=True)
        
        # Convert prices to float to ensure they are numeric
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna()  # Remove any rows where conversion failed
        
        results = {}
        
        # Weekly resampling for better decomposition
        try:
            price_series = df['Price'].resample('W').mean()
            price_series = price_series.dropna()
            print(f"Weekly resampled series length: {len(price_series)}")
        except Exception as e:
            print(f"Error in resampling: {e}")
            # Fallback - use original series
            price_series = df['Price']
            print(f"Using original series. Length: {len(price_series)}")
        
        # Check if we have enough data
        if len(price_series) < 5:  # Need at least some data
            print("Not enough data for time series analysis")
            results['error'] = "Not enough data for time series analysis"
            
            # Create blank plots with messages
            for plot_name in ['decomposition_plot', 'rolling_stats_plot', 'diff_plot', 'acf_pacf_plot']:
                fig = plt.figure(figsize=(12, 6))
                plt.text(0.5, 0.5, f'Insufficient data for {plot_name.replace("_", " ")}', 
                        horizontalalignment='center', verticalalignment='center', fontsize=14)
                plt.axis('off')
                results[plot_name] = plot_to_base64(fig)
                plt.close(fig)
                
            return jsonify({'status': 'warning', 'results': results})
        
        # Use our custom decomposition function instead of seasonal_decompose
        try:
            decomposition = custom_decompose(price_series)
            
            # Plot decomposition
            fig = plt.figure(figsize=(14, 10))
            ax1 = plt.subplot(411)
            ax1.plot(decomposition['observed'])
            ax1.set_title('Observed')
            ax1.tick_params(labelbottom=False)
            
            ax2 = plt.subplot(412)
            ax2.plot(decomposition['trend'])
            ax2.set_title('Trend')
            ax2.tick_params(labelbottom=False)
            
            ax3 = plt.subplot(413)
            ax3.plot(decomposition['seasonal'])
            ax3.set_title(f'Seasonal (Period: {decomposition["period"]})')
            ax3.tick_params(labelbottom=False)
            
            ax4 = plt.subplot(414)
            ax4.plot(decomposition['resid'])
            ax4.set_title('Residual')
            
            plt.tight_layout()
            plt.suptitle(f'Time Series Decomposition ({decomposition["model_type"].title()} Model)', 
                        fontsize=14, y=1.02)
            
            results['decomposition_plot'] = plot_to_base64(fig)
            plt.close(fig)
            print("Saved custom decomposition plot")
            
        except Exception as e:
            print(f"Custom decomposition failed: {e}")
            # Create a simplified trend visualization as fallback
            try:
                fig = plt.figure(figsize=(14, 10))
                
                # Plot original series
                ax1 = plt.subplot(211)
                ax1.plot(price_series, label='Original Price Series')
                
                # Add simple moving averages as trend indicators
                window_small = max(3, len(price_series) // 10)
                window_large = max(7, len(price_series) // 5)
                
                ma_small = price_series.rolling(window=window_small).mean()
                ma_large = price_series.rolling(window=window_large).mean()
                
                ax1.plot(ma_small, label=f'{window_small}-Period Moving Average', color='red')
                ax1.plot(ma_large, label=f'{window_large}-Period Moving Average', color='green')
                ax1.legend()
                ax1.set_title('Original Series with Moving Averages')
                
                # Plot percentage changes
                ax2 = plt.subplot(212)
                pct_change = price_series.pct_change() * 100
                ax2.plot(pct_change, label='Percentage Change', color='blue')
                ax2.set_title('Percentage Change (as alternative to residuals)')
                ax2.axhline(y=0, color='grey', linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                plt.suptitle('Simplified Time Series Analysis', fontsize=14, y=1.02)
                
                results['decomposition_plot'] = plot_to_base64(fig)
                plt.close(fig)
                print("Saved simplified trend analysis as fallback")
                
            except Exception as e2:
                print(f"Simplified trend analysis also failed: {e2}")
                # Create a blank plot with an error message as last resort
                fig = plt.figure(figsize=(14, 10))
                plt.text(0.5, 0.5, f'Decomposition not available: {str(e)}\nFallback also failed: {str(e2)}',
                        horizontalalignment='center', verticalalignment='center', fontsize=14)
                results['decomposition_plot'] = plot_to_base64(fig)
                plt.close(fig)
        
        # Calculate rolling statistics
        try:
            window = min(12, len(df) // 8)  # Adjust window size for smaller datasets
            if window < 2:
                window = 2
                
            rolling_mean = df['Price'].rolling(window=window).mean()
            rolling_std = df['Price'].rolling(window=window).std()
            
            # Plot rolling statistics
            fig = plt.figure(figsize=(14, 7))
            plt.plot(df.index, df['Price'], color='blue', label='Original')
            plt.plot(rolling_mean.index, rolling_mean, color='red', label=f'{window}-Period Rolling Mean')
            plt.plot(rolling_std.index, rolling_std, color='green', label=f'{window}-Period Rolling Std')
            plt.legend(loc='best')
            plt.title('Rolling Mean & Standard Deviation')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(True)
            plt.tight_layout()
            
            results['rolling_stats_plot'] = plot_to_base64(fig)
            plt.close(fig)
            print("Saved rolling stats plot")
        except Exception as e:
            print(f"Rolling stats plot failed: {e}")
            # Create a blank plot with an error message
            fig = plt.figure(figsize=(14, 7))
            plt.text(0.5, 0.5, f'Rolling statistics not available: {str(e)}',
                    horizontalalignment='center', verticalalignment='center', fontsize=14)
            results['rolling_stats_plot'] = plot_to_base64(fig)
            plt.close(fig)
        
        # Calculate first difference
        try:
            df['Price_diff'] = df['Price'].diff()
            
            # Plot first difference
            fig = plt.figure(figsize=(14, 7))
            plt.plot(df.index[1:], df['Price_diff'].dropna(), color='blue')
            plt.title('First Difference of Price Series')
            plt.xlabel('Date')
            plt.ylabel('Price Difference')
            plt.grid(True)
            plt.tight_layout()
            
            results['diff_plot'] = plot_to_base64(fig)
            plt.close(fig)
            print("Saved diff plot")
        except Exception as e:
            print(f"Diff plot failed: {e}")
            # Create a blank plot with an error message
            fig = plt.figure(figsize=(14, 7))
            plt.text(0.5, 0.5, f'First difference plot not available: {str(e)}',
                    horizontalalignment='center', verticalalignment='center', fontsize=14)
            results['diff_plot'] = plot_to_base64(fig)
            plt.close(fig)
        
        # Perform Augmented Dickey-Fuller test on original series
        try:
            valid_data = df['Price'].dropna()
            if len(valid_data) > 10:  # Ensure we have enough data
                dftest = adfuller(valid_data, autolag='AIC')
                dfoutput = {
                    'Test Statistic': float(dftest[0]),
                    'p-value': float(dftest[1]),
                    'Lags Used': int(dftest[2]),
                    'Observations': int(dftest[3])
                }
                
                for key, value in dftest[4].items():
                    dfoutput[f'Critical Value ({key})'] = float(value)
                    
                results['adf_test'] = dfoutput
                results['is_stationary'] = bool(dftest[1] <= 0.05)
                print("ADF test completed")
            else:
                results['adf_test'] = {'message': 'Insufficient data for ADF test'}
                results['is_stationary'] = False
        except Exception as e:
            print(f"ADF test failed: {e}")
            results['adf_test'] = {'message': f'ADF test error: {str(e)}'}
            results['is_stationary'] = False
        
        # Perform Augmented Dickey-Fuller test on differenced series
        try:
            valid_diff_data = df['Price_diff'].dropna()
            if len(valid_diff_data) > 10:  # Ensure we have enough data
                dftest_diff = adfuller(valid_diff_data, autolag='AIC')
                dfoutput_diff = {
                    'Test Statistic': float(dftest_diff[0]),
                    'p-value': float(dftest_diff[1]),
                    'Lags Used': int(dftest_diff[2]),
                    'Observations': int(dftest_diff[3])
                }
                
                for key, value in dftest_diff[4].items():
                    dfoutput_diff[f'Critical Value ({key})'] = float(value)
                    
                results['adf_test_diff'] = dfoutput_diff
                results['is_diff_stationary'] = bool(dftest_diff[1] <= 0.05)
                print("ADF test on differenced series completed")
            else:
                results['adf_test_diff'] = {'message': 'Insufficient data for ADF test on differenced series'}
                results['is_diff_stationary'] = False
        except Exception as e:
            print(f"ADF test on differenced series failed: {e}")
            results['adf_test_diff'] = {'message': f'ADF test error: {str(e)}'}
            results['is_diff_stationary'] = False
        
        # Plot ACF and PACF
        try:
            valid_diff_data = df['Price_diff'].dropna()
            if len(valid_diff_data) > 10:  # Ensure we have enough data
                max_lags = min(40, len(valid_diff_data)//2)
                if max_lags < 5:
                    max_lags = 5
                    
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
                plot_acf(valid_diff_data, lags=max_lags, ax=ax1)
                ax1.set_title('Autocorrelation Function (ACF)')
                
                plot_pacf(valid_diff_data, lags=max_lags, ax=ax2, method='ywm')
                ax2.set_title('Partial Autocorrelation Function (PACF)')
                
                plt.tight_layout()
                
                # Save ACF/PACF plot
                results['acf_pacf_plot'] = plot_to_base64(fig)
                plt.close(fig)
                print("Saved ACF/PACF plot")
            else:
                # Create a blank plot with a message if ACF/PACF fails due to insufficient data
                fig = plt.figure(figsize=(14, 10))
                plt.text(0.5, 0.5, 'ACF/PACF not available: Insufficient data',
                        horizontalalignment='center', verticalalignment='center', fontsize=14)
                results['acf_pacf_plot'] = plot_to_base64(fig)
                plt.close(fig)
        except Exception as e:
            print(f"ACF/PACF plot failed: {e}")
            # Create a blank plot with an error message
            fig = plt.figure(figsize=(14, 10))
            plt.text(0.5, 0.5, f'ACF/PACF not available: {str(e)}',
                    horizontalalignment='center', verticalalignment='center', fontsize=14)
            results['acf_pacf_plot'] = plot_to_base64(fig)
            plt.close(fig)
        
        print("Time series analysis completed successfully")
        return jsonify({'status': 'success', 'results': results})
    
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error in time series analysis: {str(e)}")
        print(error_traceback)
        
        # Create error message plots as fallback
        results = {}
        for plot_name in ['decomposition_plot', 'rolling_stats_plot', 'diff_plot', 'acf_pacf_plot']:
            fig = plt.figure(figsize=(12, 6))
            plt.text(0.5, 0.5, f'Error: {str(e)}', 
                    horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.axis('off')
            results[plot_name] = plot_to_base64(fig)
            plt.close(fig)
            
        return jsonify({
            'status': 'error', 
            'message': str(e),
            'traceback': error_traceback,
            'results': results
        })