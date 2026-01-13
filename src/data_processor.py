import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import emcee
import pymc3 as pm

def load_data(file_path):
    """
    Load neutrino experiment data from a file.
    
    Parameters:
        file_path (str): Path to the data file
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    try:
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith(('.npy', '.npz')):
            data = np.load(file_path)
        else:
            raise ValueError("Unsupported file format")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def filter_and_calibrate(data, energy_min=0.1, energy_max=10.0, 
                        baseline_min=100, baseline_max=1000):
    """
    Filter and calibrate neutrino data based on experimental constraints.
    
    Parameters:
        data (pandas.DataFrame): Raw neutrino data
        energy_min (float): Minimum energy threshold
        energy_max (float): Maximum energy threshold
        baseline_min (float): Minimum baseline distance
        baseline_max (float): Maximum baseline distance
        
    Returns:
        pandas.DataFrame: Filtered and calibrated data
    """
    # Apply energy filtering
    filtered_data = data[(data['energy'] >= energy_min) & 
                         (data['energy'] <= energy_max)]
    
    # Apply baseline filtering
    filtered_data = filtered_data[(filtered_data['baseline'] >= baseline_min) & 
                                  (filtered_data['baseline'] <= baseline_max)]
    
    # Apply basic calibration (example: normalize flux)
    if 'flux' in filtered_data.columns:
        filtered_data['flux_calibrated'] = filtered_data['flux'] / filtered_data['flux'].max()
    
    # Apply quality cuts (example: remove outliers)
    if 'chi_squared' in filtered_data.columns:
        q75, q25 = np.percentile(filtered_data['chi_squared'], [75, 25])
        iqr = q75 - q25
        filtered_data = filtered_data[filtered_data['chi_squared'] <= q75 + 1.5*iqr]
    
    return filtered_data

class NeutrinoDataProcessor:
    """
    A class to handle raw data processing, filtering, and calibration for neutrino experiments.
    """
    
    def __init__(self, data):
        """
        Initialize the processor with raw data.
        
        Parameters:
            data (pandas.DataFrame): Raw neutrino data
        """
        self.raw_data = data
        self.processed_data = None
        
    def apply_filters(self, energy_range=(0.1, 10.0), baseline_range=(100, 1000)):
        """
        Apply energy and baseline filters to the data.
        
        Parameters:
            energy_range (tuple): Energy range (min, max)
            baseline_range (tuple): Baseline range (min, max)
        """
        self.processed_data = filter_and_calibrate(
            self.raw_data, 
            energy_min=energy_range[0], 
            energy_max=energy_range[1],
            baseline_min=baseline_range[0],
            baseline_max=baseline_range[1]
        )
        
    def calibrate_flux(self, method='normalization'):
        """
        Calibrate the flux measurements.
        
        Parameters:
            method (str): Calibration method ('normalization', 'standardization')
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Run apply_filters first.")
            
        if method == 'normalization':
            self.processed_data['flux_normalized'] = (
                self.processed_data['flux'] / self.processed_data['flux'].max()
            )
        elif method == 'standardization':
            self.processed_data['flux_standardized'] = (
                (self.processed_data['flux'] - self.processed_data['flux'].mean()) / 
                self.processed_data['flux'].std()
            )
            
    def perform_statistical_analysis(self):
        """
        Perform basic statistical analysis on the processed data.
        
        Returns:
            dict: Statistical summary
        """
        if self.processed_data is None:
            raise ValueError("No processed data available.")
            
        stats_summary = {
            'mean_energy': self.processed_data['energy'].mean(),
            'std_energy': self.processed_data['energy'].std(),
            'mean_baseline': self.processed_data['baseline'].mean(),
            'std_baseline': self.processed_data['baseline'].std(),
            'total_events': len(self.processed_data),
            'flux_range': (self.processed_data['flux'].min(), self.processed_data['flux'].max())
        }
        
        return stats_summary