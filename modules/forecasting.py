import io
import base64
import pandas as pd
import numpy as np
import matplotlib
# Set non-interactive backend to avoid Tkinter issues
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from flask import jsonify
from datetime import datetime, timedelta
import traceback

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

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

def run_forecast_model(data):
    """
    Run time series forecasting models and return results.
    
    Parameters:
        data (dict): Dictionary containing model parameters and stock data
        
    Returns:
        JSON response with forecasting results
    """
    try:
        print(f"Starting forecasting with data keys: {list(data.keys())}")
        
        # Check required parameters
        required_keys = ['ticker', 'dates', 'prices', 'forecast_days', 'model_type']
        for key in required_keys:
            if key not in data:
                return jsonify({
                    'status': 'error', 
                    'message': f'Missing required parameter: {key}'
                })
        
        ticker = data['ticker']
        
        # Ensure we have dates in datetime format
        try:
            dates = pd.to_datetime(data['dates'])
        except Exception as e:
            return jsonify({
                'status': 'error', 
                'message': f'Invalid dates format: {str(e)}'
            })
        
        # Handle nested prices data structure
        try:
            raw_prices = data['prices']
            print(f"Raw prices type: {type(raw_prices)}, sample: {raw_prices[:3] if len(raw_prices) > 3 else raw_prices}")
            
            # Flatten nested lists and convert to float
            prices = []
            for price_item in raw_prices:
                if isinstance(price_item, (list, tuple)):
                    # If it's a nested structure, take the first element
                    if len(price_item) > 0:
                        prices.append(float(price_item[0]))
                    else:
                        prices.append(None)  # Handle empty nested lists
                elif price_item is not None:
                    # If it's already a number or string, convert directly
                    prices.append(float(price_item))
                else:
                    prices.append(None)  # Handle None values
                    
            # Filter out None values
            prices = [p for p in prices if p is not None and not np.isnan(p)]
            
            if len(prices) == 0:
                return jsonify({
                    'status': 'error', 
                    'message': 'No valid price data found after processing'
                })
                
            print(f"Processed prices length: {len(prices)}, sample: {prices[:3]}")
            
        except Exception as e:
            return jsonify({
                'status': 'error', 
                'message': f'Invalid prices format: {str(e)}'
            })
        
        # Parse other parameters
        try:
            forecast_days = int(data['forecast_days'])
            model_type = data['model_type']
        except Exception as e:
            return jsonify({
                'status': 'error', 
                'message': f'Invalid parameter format: {str(e)}'
            })
        
        # Create a DataFrame with valid data only
        # Ensure dates and prices have the same length
        min_length = min(len(dates), len(prices))
        valid_dates = dates[:min_length]
        valid_prices = prices[:min_length]
        
        df_data = {'Price': []}
        final_dates = []
        
        for i, (date, price) in enumerate(zip(valid_dates, valid_prices)):
            try:
                price_val = float(price)
                if not np.isnan(price_val) and price_val > 0:
                    df_data['Price'].append(price_val)
                    final_dates.append(date)
            except:
                # Skip invalid prices
                pass
        
        # Create DataFrame with valid data
        df = pd.DataFrame(df_data, index=final_dates)
        
        # Check if we have enough data
        min_required = 20  # Absolute minimum requirement for any meaningful forecast
        recommended = 60   # Recommended minimum for reliable forecasts
        
        if len(df) == 0:
            # No valid data points
            return jsonify({
                'status': 'error', 
                'message': f'No valid price data found. Please check your input data.'
            })
        elif len(df) < min_required:
            # Create a simple forecast with warning
            return simple_forecast(ticker, df, forecast_days, 
                                  warning=f'Very limited data: only {len(df)} valid data points available. ' +
                                         f'At least {recommended} points recommended for reliable forecasts.')
        elif len(df) < recommended:
            # Create forecast but with a warning about reliability
            results = simple_forecast(ticker, df, forecast_days, 
                                     warning=f'Limited data: only {len(df)} valid data points available. ' +
                                            f'At least {recommended} points recommended for reliable forecasts.')
            
            # If we have at least the minimum required, try the selected model as well
            if model_type == 'arima' and len(df) >= 30:
                try:
                    arima_results = run_arima_model(ticker, df, df[:int(len(df)*0.7)], df[int(len(df)*0.7):], 
                                                   forecast_days, limited_data=True)
                    # Replace some of the simple forecast results with ARIMA results
                    results['forecast_plot'] = arima_results['forecast_plot']
                    results['forecast_values'] = arima_results['forecast_values']
                    results['model_info'] = arima_results['model_info']
                    if 'lower_bound' in arima_results and 'upper_bound' in arima_results:
                        results['lower_bound'] = arima_results['lower_bound']
                        results['upper_bound'] = arima_results['upper_bound']
                except:
                    # Keep the simple forecast if ARIMA fails
                    pass
                    
            return jsonify({'status': 'warning', 'results': results})
        
        # If we have enough data, proceed with the selected model
        # Train-test split
        train_size = max(len(df) - 30, int(len(df) * 0.8))  # Use either last 30 days or 20% for test
        train_data = df[:train_size]
        test_data = df[train_size:]
        
        print(f"Using {len(df)} data points for forecasting with {model_type} model")
        print(f"Training data size: {len(train_data)}, Testing data size: {len(test_data)}")
        
        results = {}
        
        # Select the appropriate model
        if model_type == 'arima':
            results = run_arima_model(ticker, df, train_data, test_data, forecast_days)
        elif model_type == 'prophet':
            results = run_prophet_model(ticker, df, train_data, test_data, forecast_days)
        elif model_type == 'lstm':
            results = run_lstm_model(ticker, df, train_data, test_data, forecast_days)
        else:
            return jsonify({'status': 'error', 'message': f'Unknown model type: {model_type}'})
        
        return jsonify({'status': 'success', 'results': results})
    
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error in forecasting: {str(e)}")
        print(error_traceback)
        
        return jsonify({
            'status': 'error', 
            'message': str(e), 
            'traceback': error_traceback
        })

def simple_forecast(ticker, df, forecast_days, warning=None):
    """
    Create a simple forecast for cases with limited data.
    
    Parameters:
        ticker (str): Stock ticker symbol
        df (DataFrame): Available data
        forecast_days (int): Number of days to forecast
        warning (str): Warning message about data limitations
        
    Returns:
        dict: Results dictionary
    """
    results = {}
    
    if warning:
        results['warning'] = warning
    
    try:
        # Calculate simple statistics to base the forecast on
        latest_price = float(df['Price'].iloc[-1])
        mean_price = float(df['Price'].mean())
        
        # Calculate average daily change
        if len(df) > 5:
            daily_changes = df['Price'].pct_change().dropna()
            mean_change = float(daily_changes.mean())
            std_change = float(daily_changes.std())
        else:
            # Not enough data for meaningful change calculation
            mean_change = 0
            std_change = 0.01  # Small default value
        
        # Create forecast dates
        last_date = df.index[-1]
        future_dates = []
        for i in range(1, forecast_days + 1):
            # Use business days if we can
            try:
                next_date = last_date + pd.Timedelta(days=i)
                future_dates.append(next_date)
            except:
                # Fallback to simple string dates
                next_date = datetime.strptime(str(last_date)[:10], '%Y-%m-%d') + timedelta(days=i)
                future_dates.append(next_date)
        
        # Generate simple forecast
        forecast_values = []
        lower_bound = []
        upper_bound = []
        
        # Use naive forecast with small random walk component
        np.random.seed(42)  # For reproducibility
        
        for i in range(forecast_days):
            # Each forecast day, apply the average percentage change
            if i == 0:
                next_price = latest_price * (1 + mean_change)
            else:
                next_price = forecast_values[-1] * (1 + mean_change + np.random.normal(0, std_change/2))
                
            forecast_values.append(float(next_price))
            
            # Create confidence intervals
            margin = next_price * std_change * 1.96 * np.sqrt(i + 1)  # Widens with time
            lower_bound.append(float(max(0, next_price - margin)))
            upper_bound.append(float(next_price + margin))
        
        # Create plots
        # Forecast plot
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df.index[-30:] if len(df) > 30 else df.index, 
                df['Price'][-30:] if len(df) > 30 else df['Price'], 
                label='Historical Data')
        plt.plot(future_dates, forecast_values, 'r--', label='Simple Forecast')
        plt.fill_between(future_dates, lower_bound, upper_bound, color='red', alpha=0.2, 
                        label='95% Confidence Interval')
        plt.title(f'{ticker} Simple Price Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add warning text if applicable
        if warning:
            plt.figtext(0.5, 0.01, warning, ha='center', fontsize=10, 
                       bbox={'facecolor':'orange', 'alpha':0.2, 'pad':5})
            
        plt.tight_layout()
        results['forecast_plot'] = plot_to_base64(fig)
        plt.close(fig)
        
        # Create a simple evaluation plot
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Price'], label='Historical Price')
        
        # Add moving average if we have enough data
        if len(df) >= 7:
            window = min(7, len(df) // 3)
            ma = df['Price'].rolling(window=window).mean()
            plt.plot(df.index, ma, label=f'{window}-day Moving Average')
            
        plt.title(f'{ticker} Historical Price Data')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if warning:
            plt.figtext(0.5, 0.01, warning, ha='center', fontsize=10, 
                       bbox={'facecolor':'orange', 'alpha':0.2, 'pad':5})
            
        plt.tight_layout()
        results['backtest_plot'] = plot_to_base64(fig)
        plt.close(fig)
        
        # Return results
        results['forecast_values'] = forecast_values
        results['lower_bound'] = lower_bound
        results['upper_bound'] = upper_bound
        results['future_dates'] = [d.strftime('%Y-%m-%d') for d in future_dates]
        results['metrics'] = {'mse': None, 'rmse': None, 'mae': None}
        results['model_info'] = {'type': 'Simple Projection', 
                                'limited_data': True,
                                'data_points': len(df)}
                                
        return results
        
    except Exception as e:
        print(f"Error in simple forecast: {e}")
        # Create blank plots with error message
        for plot_name in ['forecast_plot', 'backtest_plot']:
            fig = plt.figure(figsize=(12, 6))
            plt.text(0.5, 0.5, f'Unable to generate forecast: {str(e)}',
                    horizontalalignment='center', verticalalignment='center', fontsize=14)
            if warning:
                plt.figtext(0.5, 0.3, warning, ha='center', fontsize=12, 
                           bbox={'facecolor':'orange', 'alpha':0.2, 'pad':5})
            plt.axis('off')
            results[plot_name] = plot_to_base64(fig)
            plt.close(fig)
            
        # Return basic results
        results['forecast_values'] = [df['Price'].iloc[-1]] * forecast_days if not df.empty else [0] * forecast_days
        results['lower_bound'] = [0] * forecast_days
        results['upper_bound'] = [0] * forecast_days
        results['future_dates'] = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') 
                                 for i in range(1, forecast_days + 1)]
        results['metrics'] = {'mse': None, 'rmse': None, 'mae': None}
        results['model_info'] = {'type': 'Error', 'message': str(e)}
        results['error'] = str(e)
        
        return results

def run_arima_model(ticker, df, train_data, test_data, forecast_days, limited_data=False):
    """
    Run ARIMA model for forecasting.
    
    Parameters:
        ticker (str): Stock ticker symbol
        df (DataFrame): Full dataset
        train_data (DataFrame): Training data
        test_data (DataFrame): Test data
        forecast_days (int): Number of days to forecast
        limited_data (bool): Whether we're working with limited data
        
    Returns:
        dict: Results dictionary
    """
    try:
        results = {}
        
        if limited_data:
            results['warning'] = f"Limited data: only {len(df)} valid data points available. " + \
                               f"Forecast may be less reliable."
        
        # For small datasets, use simpler model selection
        if len(df) < 100:
            # Try a few simple ARIMA models and pick the best
            best_aic = float('inf')
            best_order = (1, 1, 0)  # Default to a simple model
            
            # Candidate models
            orders = [(1,1,0), (0,1,1), (1,1,1), (2,1,0), (0,1,2)]
            
            for order in orders:
                try:
                    model = ARIMA(train_data['Price'], order=order)
                    model_fit = model.fit()
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = order
                except:
                    # Skip if this model fails
                    continue
                    
            p, d, q = best_order
            
        else:
            # Auto ARIMA for larger datasets
            try:
                auto_model = auto_arima(
                    train_data['Price'],
                    start_p=0, start_q=0,
                    test='adf',
                    max_p=3, max_q=3,  # Reduced complexity
                    max_d=1,  # Limit differencing
                    m=1,
                    seasonal=False,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True,
                    max_order=5  # Limit total order
                )
                p, d, q = auto_model.order
            except:
                # Fallback to a reasonable default
                p, d, q = (1, 1, 1)
        
        # Fit ARIMA model
        arima_model = ARIMA(train_data['Price'], order=(p, d, q))
        arima_fit = arima_model.fit()
        
        # Forecast test period if we have test data
        if len(test_data) > 0:
            test_forecast = arima_fit.forecast(steps=len(test_data))
            
            # Calculate metrics if we have enough test data
            if len(test_data) >= 5:
                test_actual = test_data['Price'].values[:len(test_forecast)]
                test_forecast_values = test_forecast[:len(test_actual)]
                
                mse = mean_squared_error(test_actual, test_forecast_values)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(test_actual, test_forecast_values)
            else:
                mse = rmse = mae = np.nan
        else:
            test_forecast = []
            mse = rmse = mae = np.nan
        
        # Fit on full data and forecast future
        final_model = ARIMA(df['Price'], order=(p, d, q))
        final_fit = final_model.fit()
        future_forecast = final_fit.forecast(steps=forecast_days)
        
        # Create confidence intervals
        forecast_result = final_fit.get_forecast(steps=forecast_days)
        conf_int = forecast_result.conf_int()
        
        # Create future dates
        future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), 
                                   periods=forecast_days, freq='B')
        
        # Ensure we have the correct number of future dates
        if len(future_dates) < forecast_days:
            # Extend with calendar days if needed
            additional_dates = pd.date_range(future_dates[-1] + pd.Timedelta(days=1), 
                                          periods=forecast_days - len(future_dates))
            future_dates = pd.DatetimeIndex(list(future_dates) + list(additional_dates))
        
        # Trim to exact length needed
        future_dates = future_dates[:forecast_days]
        
        # Plot backtest results if we have test data
        if len(test_data) > 0:
            plt.figure(figsize=(14, 7))
            plt.plot(train_data.index[-60:] if len(train_data) > 60 else train_data.index, 
                    train_data['Price'][-60:] if len(train_data) > 60 else train_data['Price'], 
                    label='Training Data', alpha=0.7)
            plt.plot(test_data.index, test_data['Price'], label='Actual Price')
            
            # Only plot test forecast if we actually have values
            if len(test_forecast) > 0:
                plt.plot(test_data.index[:len(test_forecast)], test_forecast, 
                        label=f'ARIMA({p},{d},{q}) Forecast', color='red')
                
            plt.title(f'{ticker} Stock Price Forecast using ARIMA({p},{d},{q})')
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.legend()
            plt.grid(True)
            
            # Add warning message if applicable
            if limited_data:
                plt.figtext(0.5, 0.01, results['warning'], ha='center', fontsize=10, 
                           bbox={'facecolor':'orange', 'alpha':0.2, 'pad':5})
            
            # Save backtest plot
            fig = plt.gcf()
            results['backtest_plot'] = plot_to_base64(fig)
            plt.close(fig)
        else:
            # Create a simple historical plot if no test data
            plt.figure(figsize=(14, 7))
            plt.plot(df.index, df['Price'], label='Historical Data')
            plt.title(f'{ticker} Historical Stock Price')
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.legend()
            plt.grid(True)
            
            if limited_data:
                plt.figtext(0.5, 0.01, results['warning'], ha='center', fontsize=10, 
                           bbox={'facecolor':'orange', 'alpha':0.2, 'pad':5})
            
            fig = plt.gcf()
            results['backtest_plot'] = plot_to_base64(fig)
            plt.close(fig)
        
        # Plot future forecast
        plt.figure(figsize=(14, 7))
        
        # Plot historical data - use last 90 days or all if less
        plt.plot(df.index[-90:] if len(df) > 90 else df.index, 
                df['Price'][-90:] if len(df) > 90 else df['Price'], 
                label='Historical Data')
                
        plt.plot(future_dates, future_forecast, label=f'ARIMA({p},{d},{q}) Forecast', 
                color='red', linestyle='--')
        
        # Only plot confidence intervals if they are valid
        if not conf_int.empty and not conf_int.isnull().all().all() and len(conf_int) == len(future_dates):
            plt.fill_between(future_dates,
                            conf_int.iloc[:, 0].values,
                            conf_int.iloc[:, 1].values,
                            color='red', alpha=0.2, label='95% Confidence Interval')
        
        plt.axvline(x=df.index[-1], color='black', linestyle='-', alpha=0.7)
        plt.title(f'{ticker} Stock Price Forecast for the Next {forecast_days} Trading Days')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        
        if limited_data:
            plt.figtext(0.5, 0.01, results['warning'], ha='center', fontsize=10, 
                       bbox={'facecolor':'orange', 'alpha':0.2, 'pad':5})
        
        # Save forecast plot
        fig = plt.gcf()
        results['forecast_plot'] = plot_to_base64(fig)
        plt.close(fig)
        
        # Return results - convert to list properly
        results['forecast_values'] = future_forecast.tolist() if hasattr(future_forecast, 'tolist') else future_forecast.values.tolist()
        
        # Handle confidence intervals
        if not conf_int.empty and not conf_int.isnull().all().all() and len(conf_int) == len(future_dates):
            results['lower_bound'] = conf_int.iloc[:, 0].values.tolist()
            results['upper_bound'] = conf_int.iloc[:, 1].values.tolist()
        else:
            # Fallback to simple percentage-based confidence interval
            volatility = df['Price'].pct_change().std() * 1.96  # 95% confidence
            results['lower_bound'] = [max(0, x * (1 - volatility)) for x in future_forecast]
            results['upper_bound'] = [x * (1 + volatility) for x in future_forecast]
        
        results['future_dates'] = [d.strftime('%Y-%m-%d') for d in future_dates[:forecast_days]]
        results['metrics'] = {
            'mse': float(mse) if not np.isnan(mse) else None, 
            'rmse': float(rmse) if not np.isnan(rmse) else None, 
            'mae': float(mae) if not np.isnan(mae) else None
        }
        results['model_info'] = {
            'type': 'ARIMA', 
            'order': f'({p},{d},{q})',
            'data_points': len(df)
        }
        
        if limited_data:
            results['model_info']['limited_data'] = True
        
        return results
    except Exception as e:
        print(f"ARIMA error: {e}")
        # If ARIMA fails, return a fallback simple moving average forecast
        return simple_forecast(ticker, df, forecast_days, 
                              warning=f"ARIMA model failed: {str(e)}. Using simple forecast instead.")

def run_prophet_model(ticker, df, train_data, test_data, forecast_days):
    """
    Run Facebook Prophet model for forecasting.
    
    Parameters:
        ticker (str): Stock ticker symbol
        df (DataFrame): Full dataset
        train_data (DataFrame): Training data
        test_data (DataFrame): Test data
        forecast_days (int): Number of days to forecast
        
    Returns:
        dict: Results dictionary
    """
    try:
        results = {}
        
        # Check if we have enough data for Prophet
        if len(train_data) < 30:
            return simple_forecast(ticker, df, forecast_days, 
                                  warning="Insufficient data for Prophet model. Using simple forecast instead.")
        
        # Prepare data for Prophet
        prophet_train = train_data.reset_index()
        prophet_train = prophet_train.rename(columns={'index': 'ds', 'Price': 'y'})
        
        # Create and fit Prophet model - with simplified parameters for stability
        prophet_model = Prophet(
            daily_seasonality=False,  # Simplified
            weekly_seasonality=True, 
            yearly_seasonality=True if len(train_data) > 365 else False,
            changepoint_prior_scale=0.05  # Default value
        )
        prophet_model.fit(prophet_train)
        
        # Test period forecast
        if len(test_data) > 0:
            future_test = prophet_model.make_future_dataframe(periods=len(test_data))
            forecast_test = prophet_model.predict(future_test)
            test_forecast = forecast_test.tail(len(test_data))['yhat'].values
            
            # Calculate metrics if we have enough test data
            if len(test_data) >= 5:
                test_actual = test_data['Price'].values[:len(test_forecast)]
                test_forecast = test_forecast[:len(test_actual)]
                
                mse = mean_squared_error(test_actual, test_forecast)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(test_actual, test_forecast)
            else:
                mse = rmse = mae = np.nan
        else:
            test_forecast = []
            mse = rmse = mae = np.nan
        
        # Fit on full data and forecast future
        prophet_full = df.reset_index()
        prophet_full = prophet_full.rename(columns={'index': 'ds', 'Price': 'y'})
        
        final_model = Prophet(
            daily_seasonality=False,  # Simplified
            weekly_seasonality=True, 
            yearly_seasonality=True if len(df) > 365 else False,
            changepoint_prior_scale=0.05  # Default value
        )
        final_model.fit(prophet_full)
        
        future = final_model.make_future_dataframe(periods=forecast_days)
        forecast = final_model.predict(future)
        
        # Extract forecast values for the future periods
        future_forecast = forecast.tail(forecast_days)['yhat'].values
        lower_bound = forecast.tail(forecast_days)['yhat_lower'].values
        upper_bound = forecast.tail(forecast_days)['yhat_upper'].values
        future_dates = pd.to_datetime(forecast.tail(forecast_days)['ds'].dt.date)
        
        # Plot components if we have enough data
        if len(df) > 60:
            try:
                fig_comp = final_model.plot_components(forecast)
                results['components_plot'] = plot_to_base64(fig_comp)
                plt.close(fig_comp)
            except Exception as e:
                print(f"Prophet components plot failed: {e}")
                # Create a dummy components plot
                fig = plt.figure(figsize=(14, 10))
                plt.text(0.5, 0.5, f'Components visualization not available: {str(e)}',
                        horizontalalignment='center', verticalalignment='center', fontsize=14)
                results['components_plot'] = plot_to_base64(fig)
                plt.close(fig)
        
        # Plot backtest results
        if len(test_data) > 0:
            plt.figure(figsize=(14, 7))
            plt.plot(train_data.index[-60:] if len(train_data) > 60 else train_data.index, 
                    train_data['Price'][-60:] if len(train_data) > 60 else train_data['Price'], 
                    label='Training Data', alpha=0.7)
            plt.plot(test_data.index, test_data['Price'], label='Actual Price')
            
            # Only plot test forecast if we have data
            if len(test_forecast) > 0:
                plt.plot(test_data.index[:len(test_forecast)], test_forecast, 
                        label='Prophet Forecast', color='green')
                        
            plt.title(f'{ticker} Stock Price Forecast using Prophet')
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.legend()
            plt.grid(True)
            
            # Save backtest plot
            fig = plt.gcf()
            results['backtest_plot'] = plot_to_base64(fig)
            plt.close(fig)
        else:
            # Simple historical plot if no test data
            plt.figure(figsize=(14, 7))
            plt.plot(df.index, df['Price'], label='Historical Data')
            plt.title(f'{ticker} Historical Stock Price')
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.legend()
            plt.grid(True)
            
            fig = plt.gcf()
            results['backtest_plot'] = plot_to_base64(fig)
            plt.close(fig)
        
        # Create custom forecast plot
        plt.figure(figsize=(14, 7))
        plt.plot(df.index[-90:] if len(df) > 90 else df.index, 
                df['Price'][-90:] if len(df) > 90 else df['Price'], 
                label='Historical Data')
        
        plt.plot(future_dates, future_forecast, label='Prophet Forecast', 
                color='green', linestyle='--')
        plt.fill_between(future_dates, lower_bound, upper_bound,
                        color='green', alpha=0.2, label='95% Confidence Interval')
        
        plt.axvline(x=df.index[-1], color='black', linestyle='-', alpha=0.7)
        plt.title(f'{ticker} Stock Price Forecast using Prophet')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        
        # Save forecast plot
        fig = plt.gcf()
        results['forecast_plot'] = plot_to_base64(fig)
        plt.close(fig)
        
        # Return results - convert NumPy arrays to lists
        results['forecast_values'] = future_forecast.tolist()
        results['lower_bound'] = lower_bound.tolist()
        results['upper_bound'] = upper_bound.tolist()
        results['future_dates'] = [d.strftime('%Y-%m-%d') for d in future_dates]
        results['metrics'] = {
            'mse': float(mse) if not np.isnan(mse) else None, 
            'rmse': float(rmse) if not np.isnan(rmse) else None, 
            'mae': float(mae) if not np.isnan(mae) else None
        }
        results['model_info'] = {
            'type': 'Prophet',
            'data_points': len(df)
        }
        
        return results
        
    except Exception as e:
        print(f"Prophet error: {e}")
        # If Prophet fails, return a simple forecast
        return simple_forecast(ticker, df, forecast_days, 
                              warning=f"Prophet model failed: {str(e)}. Using simple forecast instead.")

def run_lstm_model(ticker, df, train_data, test_data, forecast_days):
    """
    Run LSTM model for forecasting.
    
    Parameters:
        ticker (str): Stock ticker symbol
        df (DataFrame): Full dataset
        train_data (DataFrame): Training data
        test_data (DataFrame): Test data
        forecast_days (int): Number of days to forecast
        
    Returns:
        dict: Results dictionary
    """
    try:
        results = {}
        
        # Check if we have enough data for LSTM
        if len(train_data) < 60:  # LSTM needs more data
            return simple_forecast(ticker, df, forecast_days, 
                                 warning="Insufficient data for LSTM model. Using simple forecast instead.")
        
        # Create sequences for LSTM
        def create_sequence_data(data, time_steps=60):
            X, y = [], []
            for i in range(len(data) - time_steps):
                X.append(data[i:(i + time_steps)])
                y.append(data[i + time_steps])
            return np.array(X), np.array(y)
        
        # Prepare data for LSTM
        time_steps = min(60, len(train_data) // 3)  # Reduce time steps if data is limited
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        price_data = df['Price'].values.reshape(-1, 1)
        scaled_data = scaler.fit_transform(price_data)
        
        # Split into train and test sets
        train_size = len(train_data)
        train_scaled = scaled_data[:train_size]
        
        # Only create test sequences if we have test data
        if len(test_data) > 0:
            test_scaled = scaled_data[train_size - time_steps:, :]
        
        # Create sequences for training
        X_train, y_train = create_sequence_data(train_scaled, time_steps)
        
        # Check if sequences were created successfully
        if len(X_train) == 0:
            return simple_forecast(ticker, df, forecast_days, 
                                  warning="Could not create sequence data for LSTM. Using simple forecast instead.")
        
        # Create sequences for testing if available
        if len(test_data) > 0:
            X_test, y_test = create_sequence_data(
                scaled_data[train_size - time_steps:train_size + len(test_data)], 
                time_steps
            )
        
        # Reshape for LSTM [samples, time steps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        
        # Reshape test data if available
        if len(test_data) > 0 and 'X_test' in locals() and len(X_test) > 0:
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Build LSTM model - simplified for smaller datasets
        lstm_model = Sequential()
        
        # Use a simpler architecture for smaller datasets
        if len(train_data) < 200:
            lstm_model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
            lstm_model.add(Dropout(0.2))
            lstm_model.add(Dense(units=1))
        else:
            # More complex model for larger datasets
            lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            lstm_model.add(Dropout(0.2))
            lstm_model.add(LSTM(units=50, return_sequences=False))
            lstm_model.add(Dropout(0.2))
            lstm_model.add(Dense(units=1))
        
        # Compile the model
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train the model - reduce epochs for smaller datasets
        epochs = 10 if len(train_data) < 200 else 20
        batch_size = min(32, len(X_train) // 4)  # Smaller batch size for smaller datasets
        
        lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        
        # Make predictions on test data if available
        if len(test_data) > 0 and 'X_test' in locals() and len(X_test) > 0:
            lstm_predictions = lstm_model.predict(X_test)
            
            # Inverse transform to get actual price values
            lstm_predictions = scaler.inverse_transform(lstm_predictions)
            
            if 'y_test' in locals() and len(y_test) > 0:
                y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
                
                # Calculate metrics
                if len(y_test_actual) > 0 and len(lstm_predictions) > 0:
                    mse = mean_squared_error(y_test_actual, lstm_predictions)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test_actual, lstm_predictions)
                else:
                    mse = rmse = mae = np.nan
            else:
                mse = rmse = mae = np.nan
        else:
            lstm_predictions = []
            mse = rmse = mae = np.nan
        
        # Prepare for future forecasting
        # Use the last time_steps days to predict future
        last_sequence = scaled_data[-time_steps:].reshape(1, time_steps, 1)
        future_forecast = []
        
        current_batch = np.copy(last_sequence)  # Create a deep copy to avoid modifying the original
        for i in range(forecast_days):
            # Get prediction for next day
            current_pred = lstm_model.predict(current_batch)[0]
            future_forecast.append(current_pred[0])
            
            # Update the sequence by removing first value and adding prediction at the end
            current_batch = np.roll(current_batch, -1, axis=1)
            current_batch[0, -1, 0] = current_pred[0]
        
        # Inverse transform predictions
        future_forecast_array = np.array(future_forecast).reshape(-1, 1)
        future_forecast = scaler.inverse_transform(future_forecast_array).flatten()
        
        # Calculate confidence interval based on historical volatility
        hist_volatility = df['Price'].pct_change().std()
        lower_bound = future_forecast * (1 - 1.96 * hist_volatility)
        upper_bound = future_forecast * (1 + 1.96 * hist_volatility)
        
        # Create future dates
        future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), 
                                    periods=forecast_days, freq='B')
        
        # Ensure we have the correct number of future dates
        if len(future_dates) < forecast_days:
            # Extend with calendar days if needed
            additional_dates = pd.date_range(future_dates[-1] + pd.Timedelta(days=1), 
                                           periods=forecast_days - len(future_dates))
            future_dates = pd.DatetimeIndex(list(future_dates) + list(additional_dates))
            
        future_dates = future_dates[:forecast_days]  # Ensure exactly forecast_days
        
        # Plot forecasts
        plt.figure(figsize=(14, 7))
        plt.plot(df.index[-90:] if len(df) > 90 else df.index, 
                df['Price'][-90:] if len(df) > 90 else df['Price'], 
                label='Historical Data')
        plt.plot(future_dates, future_forecast, label='LSTM Forecast', color='purple', linestyle='--')
        plt.fill_between(future_dates,
                        lower_bound,
                        upper_bound,
                        color='purple', alpha=0.2, label='95% Confidence Interval')
        plt.axvline(x=df.index[-1], color='black', linestyle='-', alpha=0.7)
        plt.title(f'{ticker} Stock Price Forecast using LSTM')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        
        # Save forecast plot
        fig = plt.gcf()
        results['forecast_plot'] = plot_to_base64(fig)
        plt.close(fig)
        
        # Plot test predictions if we have test data
        if len(test_data) > 0 and len(lstm_predictions) > 0:
            plt.figure(figsize=(14, 7))
            plt.plot(train_data.index[-60:] if len(train_data) > 60 else train_data.index, 
                    train_data['Price'][-60:] if len(train_data) > 60 else train_data['Price'], 
                    label='Training Data', alpha=0.7)
            plt.plot(test_data.index, test_data['Price'], label='Actual Price')
            
            # Create dates for test predictions
            if len(y_test_actual) > 0:
                pred_dates = test_data.index[:len(lstm_predictions)]
                
                # Adjust if lengths don't match
                if len(pred_dates) > len(lstm_predictions):
                    pred_dates = pred_dates[:len(lstm_predictions)]
                elif len(pred_dates) < len(lstm_predictions):
                    lstm_predictions = lstm_predictions[:len(pred_dates)]
                    
                plt.plot(pred_dates, lstm_predictions, label='LSTM Predictions', color='purple')
                
            plt.title(f'{ticker} Stock Price LSTM Model Validation')
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.legend()
            plt.grid(True)
            
            # Save backtest plot
            fig = plt.gcf()
            results['backtest_plot'] = plot_to_base64(fig)
            plt.close(fig)
        else:
            # Simple historical plot if no test predictions
            plt.figure(figsize=(14, 7))
            plt.plot(df.index, df['Price'], label='Historical Data')
            plt.title(f'{ticker} Historical Stock Price')
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.legend()
            plt.grid(True)
            
            fig = plt.gcf()
            results['backtest_plot'] = plot_to_base64(fig)
            plt.close(fig)
        
        # Return results - ensure we're returning lists not NumPy arrays
        results['forecast_values'] = future_forecast.tolist()
        results['lower_bound'] = lower_bound.tolist()
        results['upper_bound'] = upper_bound.tolist()
        results['future_dates'] = [d.strftime('%Y-%m-%d') for d in future_dates[:forecast_days]]
        results['metrics'] = {
            'mse': float(mse) if not np.isnan(mse) else None, 
            'rmse': float(rmse) if not np.isnan(rmse) else None, 
            'mae': float(mae) if not np.isnan(mae) else None
        }
        results['model_info'] = {
            'type': 'LSTM', 
            'lookback': time_steps,
            'data_points': len(df)
        }
        
        return results
        
    except Exception as e:
        print(f"LSTM error: {e}")
        # If LSTM fails, return a simple forecast
        return simple_forecast(ticker, df, forecast_days, 
                              warning=f"LSTM model failed: {str(e)}. Using simple forecast instead.")