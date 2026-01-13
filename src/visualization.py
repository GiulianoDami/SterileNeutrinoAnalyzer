import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

def plot_neutrino_flux(data):
    """
    Plot neutrino flux data with error bars
    
    Parameters:
    data (DataFrame): DataFrame containing 'energy' and 'flux' columns
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot flux vs energy
    ax.errorbar(data['energy'], data['flux'], yerr=data.get('flux_error', None), 
                fmt='o', capsize=5, alpha=0.7, color='blue')
    
    ax.set_xlabel('Neutrino Energy (GeV)')
    ax.set_ylabel('Flux (cm⁻² s⁻¹)')
    ax.set_title('Neutrino Flux Distribution')
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_model_comparison(models):
    """
    Compare different neutrino oscillation models
    
    Parameters:
    models (dict): Dictionary with model names as keys and model results as values
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each model
    for name, model_data in models.items():
        if isinstance(model_data, dict):
            energy = model_data.get('energy', np.linspace(1, 10, 100))
            flux = model_data.get('flux', np.ones_like(energy))
            ax.plot(energy, flux, label=name, linewidth=2)
        else:
            ax.plot(model_data.get('energy', np.linspace(1, 10, 100)), 
                   model_data.get('flux', np.ones_like(np.linspace(1, 10, 100))), 
                   label=name, linewidth=2)
    
    ax.set_xlabel('Neutrino Energy (GeV)')
    ax.set_ylabel('Flux (cm⁻² s⁻¹)')
    ax.set_title('Model Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def save_plot(filename):
    """
    Save the current plot to a file
    
    Parameters:
    filename (str): Path to save the plot
    """
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()