from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from modules.data_manager import fetch_stock_data
from modules.visualization import generate_basic_plots
from modules.time_series import perform_time_series_analysis
from modules.forecasting import run_forecast_model
from modules.auth import init_db, create_user, authenticate_user, login_required, get_current_user
import config
import os

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this-in-production')

# Initialize database
init_db()

@app.route('/')
@login_required
def home():
    """Render the home page."""
    user = get_current_user()
    return render_template('index.html', user=user)

@app.route('/login')
def login():
    """Render the login page."""
    if 'user_id' in session:
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/register')
def register():
    """Render the registration page."""
    if 'user_id' in session:
        return redirect(url_for('home'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Logout user and redirect to login page."""
    session.clear()
    return redirect(url_for('login'))

@app.route('/api/login', methods=['POST'])
def api_login():
    """Handle login API requests."""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'success': False, 'message': 'Username and password are required'})
    
    result = authenticate_user(username, password)
    
    if result['success']:
        session['user_id'] = result['user']['id']
        session['username'] = result['user']['username']
        return jsonify({'success': True, 'message': 'Login successful'})
    else:
        return jsonify(result)

@app.route('/api/register', methods=['POST'])
def api_register():
    """Handle registration API requests."""
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    if not all([username, email, password]):
        return jsonify({'success': False, 'message': 'All fields are required'})
    
    if len(password) < 6:
        return jsonify({'success': False, 'message': 'Password must be at least 6 characters long'})
    
    result = create_user(username, email, password)
    return jsonify(result)

@app.route('/fetch_stock_data', methods=['POST'])
@login_required
def fetch_data():
    """Fetch stock data based on ticker and date range."""
    return fetch_stock_data(
        request.json['ticker'],
        request.json['start_date'],
        request.json['end_date']
    )

@app.route('/generate_plots', methods=['POST'])
@login_required
def generate_plots():
    """Generate and return basic plots for the data."""
    return generate_basic_plots(request.json)

@app.route('/time_series_analysis', methods=['POST'])
@login_required
def time_series_analysis():
    """Perform time series decomposition and stationarity tests."""
    return perform_time_series_analysis(request.json)

@app.route('/run_forecasting', methods=['POST'])
@login_required
def run_forecasting():
    """Run time series forecasting models and return results."""
    return run_forecast_model(request.json)

if __name__ == '__main__':
    app.run(debug=config.DEBUG, host=config.HOST, port=config.PORT)