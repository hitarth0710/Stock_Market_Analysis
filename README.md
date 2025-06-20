# Stock Market Analysis & Forecasting Dashboard

![Dashboard Preview](![image](https://github.com/user-attachments/assets/45367fb3-517a-4829-858f-ad7e3199e09f)
)

A comprehensive web application for analyzing stock market trends and forecasting future price movements using various time series analysis techniques and machine learning models.

## Overview

This project integrates advanced stock market analysis tools with an intuitive web interface, allowing users to:
- Fetch and visualize historical stock data
- Perform time series decomposition and statistical analysis
- Generate price forecasts using multiple predictive models
- Compare model performance and accuracy metrics

## Features

### Data Analysis
- **Historical Data Visualization**: Interactive price charts with moving averages
- **Returns Analysis**: Distribution of daily returns and volatility patterns
- **Volume Analysis**: Trading volume patterns and anomaly detection
- **Statistical Metrics**: Key statistics and performance indicators

![Data Analysis Screenshot](![image](https://github.com/user-attachments/assets/d075aa0a-3a3e-456a-8df9-4acc03bdfffd)
)

### Time Series Analysis
- **Decomposition**: Breakdown of time series into trend, seasonality, and residual components
- **Stationarity Testing**: Augmented Dickey-Fuller test with visual interpretation
- **ACF & PACF**: Autocorrelation and partial autocorrelation function plots for model parameter selection

![Time Series Analysis Screenshot](![image](https://github.com/user-attachments/assets/5e856b4a-7c3d-4968-84f1-700716090ec0)
)

### Forecasting Models
- **ARIMA**: AutoRegressive Integrated Moving Average model
- **Prophet**: Facebook's time series forecasting algorithm
- **LSTM**: Long Short-Term Memory neural network model
- **Model Comparison**: Side-by-side performance metrics (RMSE, MSE, MAE)

![Forecasting Models Screenshot](![image](https://github.com/user-attachments/assets/9b21cb1c-d83c-41fa-ace0-7f765ff66c8f)
)

### User Interface
- **Responsive Design**: Works on desktop and mobile devices
- **Interactive Components**: Tabbed interface for organized data presentation
- **Secure Authentication**: User registration and login functionality
- **Visual Feedback**: Loading indicators and intuitive navigation

## Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Time Series Analysis**: StatsModels
- **Machine Learning**: Prophet, TensorFlow (LSTM)
- **Stock Data**: yfinance API

## Getting Started

### Prerequisites
- Python 3.10(recommended)
- pip package manager

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd stock-market-analysis
```

2. Install required packages
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Usage Guide

1. **Log in or register** for a new account
2. Enter a **stock ticker symbol** (e.g., AAPL for Apple Inc.)
3. Select a **date range** for analysis
4. Click **"Fetch Stock Data"** to retrieve historical information
5. Explore the **Data Overview** tab for price, returns, and volume charts
6. Run **Time Series Analysis** for deeper statistical insights
7. Select a **forecasting model** and parameters
8. Generate and view forecasts with performance metrics

## Project Structure

- `app.py`: Main Flask application file
- `modules`: Core functionality modules
  - `data_manager.py`: Stock data retrieval and processing
  - `visualization.py`: Chart generation
  - `time_series.py`: Time series analysis functions
  - `forecasting.py`: Predictive models implementation
  - `auth.py`: User authentication
- `templates`: HTML templates
- `static`: CSS, JavaScript and other static assets
- `notebook`: Jupyter notebooks for model development

## Sample Results

### ARIMA Model Performance
![ARIMA Results](![image](https://github.com/user-attachments/assets/24d3f6de-5933-49e8-88fe-7273158e9614)
)

### Prophet Model Performance
![Prophet Results](![image](https://github.com/user-attachments/assets/d29a37d5-6277-444d-923d-77b40fb4ab8e)
)

### LSTM Model Performance
![LSTM Results](![image](https://github.com/user-attachments/assets/357b6583-aaef-4a1a-a9a5-c1638ff85317)
)


## Acknowledgements

- Data provided by Yahoo Finance via the yfinance library
- Statistical models from StatsModels and Prophet libraries
- Deep learning capabilities powered by TensorFlow

## Authors

This project was developed by:

- **Hitarth Soni**
- **Bhavya Diwan**

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

- **Hitarth Soni**
  - Email: hitarthsoni947@gmail.com
  - GitHub: [hitarthsoni](https://github.com/hitarth0710)
  - LinkedIn: [Hitarth Soni](https://www.linkedin.com/in/hitarth-soni-80a18a299/)

- **Bhavya Diwan**
  - Email: bhavyadiwan1015@gmail.com
  - GitHub: [bhavyadiwan](https://github.com/diwanbhavya)
  - LinkedIn: [Bhavya Diwan](https://www.linkedin.com/in/bhavya-diwan-267382253?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
