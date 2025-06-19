import pandas as pd
import numpy as np
import yfinance as yf
from flask import jsonify
from datetime import datetime, timedelta
import traceback

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch and process stock data for a given ticker and date range.
    
    Parameters:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        JSON response with stock data or error message
    """
    try:
        print(f"Fetching data for {ticker} from {start_date} to {end_date}")
        
        # Download stock data
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        if stock_data.empty:
            return jsonify({'status': 'error', 'message': f'No data found for ticker {ticker}'})
        
        # Basic preprocessing
        if 'Adj Close' in stock_data.columns:
            stock_data = stock_data.rename(columns={'Adj Close': 'Price'})
        elif 'Close' in stock_data.columns:
            stock_data = stock_data.rename(columns={'Close': 'Price'})
        else:
            return jsonify({'status': 'error', 'message': f'Invalid data format for ticker {ticker}'})
        
        # Make sure we have all required columns
        required_columns = ['Price', 'Volume']
        for col in required_columns:
            if col not in stock_data.columns:
                stock_data[col] = 0
                
        # Calculate daily returns
        stock_data['Returns'] = stock_data['Price'].pct_change() * 100
        
        # Calculate moving averages
        stock_data['MA_50'] = stock_data['Price'].rolling(window=50).mean()
        stock_data['MA_200'] = stock_data['Price'].rolling(window=200).mean()
        
        # Add 20-day volume moving average
        stock_data['Volume_MA_20'] = stock_data['Volume'].rolling(window=20).mean()
        
        # Calculate average volume for reference
        avg_volume = float(stock_data['Volume'].mean())
        
        # Calculate statistics for the overview section
        stats = {}
        
        # Price statistics
        stats['price_min'] = float(stock_data['Price'].min())
        stats['price_max'] = float(stock_data['Price'].max())
        stats['price_mean'] = float(stock_data['Price'].mean())
        stats['price_std'] = float(stock_data['Price'].std())
        stats['price_latest'] = float(stock_data['Price'].iloc[-1])
        
        # Return statistics
        stats['return_min'] = float(stock_data['Returns'].min())
        stats['return_max'] = float(stock_data['Returns'].max())
        stats['return_mean'] = float(stock_data['Returns'].mean())
        stats['return_std'] = float(stock_data['Returns'].std())
        
        # Volume statistics
        stats['volume_min'] = float(stock_data['Volume'].min())
        stats['volume_max'] = float(stock_data['Volume'].max())
        stats['volume_mean'] = avg_volume
        stats['volume_std'] = float(stock_data['Volume'].std())
        
        # Calculate some additional metrics
        # Volatility (annualized)
        daily_volatility = stock_data['Returns'].std()
        stats['annualized_volatility'] = float(daily_volatility * np.sqrt(252))
        
        # Sharpe ratio approximation (assuming risk-free rate of 1%)
        risk_free_rate = 1.0  # 1% annual
        daily_risk_free = risk_free_rate / 252
        daily_excess_return = stock_data['Returns'].mean() - daily_risk_free
        stats['sharpe_ratio'] = float(daily_excess_return / daily_volatility * np.sqrt(252)) if daily_volatility > 0 else 0
        
        # Performance metrics
        if len(stock_data) >= 252:  # At least one year of data
            stats['yearly_return'] = float(((stock_data['Price'].iloc[-1] / stock_data['Price'].iloc[-252]) - 1) * 100)
        else:
            stats['yearly_return'] = None
            
        if len(stock_data) >= 63:  # At least one quarter of data (approx 63 trading days)
            stats['quarterly_return'] = float(((stock_data['Price'].iloc[-1] / stock_data['Price'].iloc[-63]) - 1) * 100)
        else:
            stats['quarterly_return'] = None
            
        if len(stock_data) >= 21:  # At least one month of data (approx 21 trading days)
            stats['monthly_return'] = float(((stock_data['Price'].iloc[-1] / stock_data['Price'].iloc[-21]) - 1) * 100)
        else:
            stats['monthly_return'] = None
            
        stats['weekly_return'] = float(((stock_data['Price'].iloc[-1] / stock_data['Price'].iloc[-5]) - 1) * 100) if len(stock_data) >= 5 else None
        
        # Reset index to work with dates as a column
        stock_data = stock_data.reset_index()
        
        # Make sure all columns are the right type before conversion
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        
        # Handle any missing columns for OHLC data
        for col in ['Open', 'High', 'Low']:
            if col not in stock_data.columns:
                stock_data[col] = stock_data['Price']
                
        # Convert to JSON - correctly extract lists from DataFrame columns
        result = {
            'ticker': ticker,
            'dates': stock_data['Date'].dt.strftime('%Y-%m-%d').values.tolist(),
            'prices': stock_data['Price'].astype(float).values.tolist(),
            'high': stock_data['High'].astype(float).values.tolist(),
            'low': stock_data['Low'].astype(float).values.tolist(),
            'open': stock_data['Open'].astype(float).values.tolist(),
            'volume': stock_data['Volume'].astype(float).values.tolist(),
            'volume_ma_20': stock_data['Volume_MA_20'].fillna(0).astype(float).values.tolist(),
            'avg_volume': avg_volume,
            'returns': stock_data['Returns'].fillna(0).astype(float).values.tolist(),
            'ma_50': stock_data['MA_50'].fillna(0).astype(float).values.tolist(),
            'ma_200': stock_data['MA_200'].fillna(0).astype(float).values.tolist(),
            'stats': stats,  # Add the statistics dictionary
            'status': 'success',
            'message': f'Successfully fetched data for {ticker}'
        }
        
        print(f"Data processed successfully. Keys: {list(result.keys())}")
        return jsonify(result)
    
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error fetching data: {str(e)}")
        print(error_traceback)
        
        return jsonify({
            'status': 'error', 
            'message': str(e),
            'traceback': error_traceback
        })