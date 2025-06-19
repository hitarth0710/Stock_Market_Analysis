"""
Configure matplotlib settings globally for the application
"""
import matplotlib
# Force matplotlib to use a non-interactive backend to avoid Tkinter issues
matplotlib.use('Agg')

# This module ensures matplotlib is configured before any other module imports matplotlib