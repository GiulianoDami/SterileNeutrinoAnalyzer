import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import emcee
import pymc3 as pm

def calculate_uncertainty(flux, uncertainty):
    """
    Calculate the uncertainty in flux measurements.
    
    Parameters:
    flux (array-like): Flux measurements
    uncertainty (array-like): Uncertainty values for each flux measurement
    
    Returns:
    array: Combined uncertainty values
    """
    flux = np.array(flux)
    uncertainty = np.array(uncertainty)
    
    # Calculate combined uncertainty (assuming independent errors)
    combined_uncertainty = np.sqrt(uncertainty**2 + (0.05 * flux)**2)
    
    return combined_uncertainty

def format_results(results):
    """
    Format analysis results into a readable structure.
    
    Parameters:
    results (dict): Dictionary containing analysis results
    
    Returns:
    dict: Formatted results with proper keys and values
    """
    formatted = {}
    
    # Convert numpy types to native Python types
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            formatted[key] = value.tolist()
        elif isinstance(value, np.float64):
            formatted[key] = float(value)
        elif isinstance(value, np.int64):
            formatted[key] = int(value)
        else:
            formatted[key] = value
    
    return formatted

def validate_input(data):
    """
    Validate input data for analysis.
    
    Parameters:
    data (array-like): Input data to validate
    
    Returns:
    bool: True if data is valid, False otherwise
    """
    if data is None:
        return False
    
    data = np.array(data)
    
    # Check for NaN or infinite values
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        return False
    
    # Check for non-positive values where required
    if np.any(data <= 0):
        return False
    
    # Check for reasonable data size
    if len(data) < 2:
        return False
    
    return True