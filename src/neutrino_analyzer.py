import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import pymc3 as pm
from scipy import stats
from scipy.optimize import minimize
import warnings

class NeutrinoAnalyzer:
    """
    Main analyzer class for performing neutrino oscillation analysis and sterile neutrino detection
    """
    
    def __init__(self, data_file=None):
        """
        Initialize the NeutrinoAnalyzer
        
        Parameters:
        data_file (str): Path to neutrino data file (optional)
        """
        self.data = None
        self.model_results = {}
        
        if data_file:
            self.load_data(data_file)
    
    def load_data(self, data_file):
        """
        Load neutrino data from file
        
        Parameters:
        data_file (str): Path to data file
        """
        try:
            self.data = pd.read_csv(data_file)
        except Exception as e:
            warnings.warn(f"Could not load data: {e}")
            self.data = pd.DataFrame()
    
    def calculate_oscillation_probability(self, E, L, delta_m2, theta):
        """
        Calculate neutrino oscillation probability
        
        Parameters:
        E (float): Neutrino energy (GeV)
        L (float): Baseline distance (km)
        delta_m2 (float): Mass squared difference (eV^2)
        theta (float): Mixing angle (radians)
        
        Returns:
        float: Oscillation probability
        """
        k = 1.27 * delta_m2 * L / E
        return np.sin(2 * theta)**2 * np.sin(k)**2
    
    def fit_oscillation_model(self, energies, fluxes, baseline=100):
        """
        Fit oscillation model to data
        
        Parameters:
        energies (array): Neutrino energies
        fluxes (array): Measured fluxes
        baseline (float): Baseline distance (km)
        
        Returns:
        dict: Fitted parameters
        """
        # Simple linear fit for demonstration
        log_energies = np.log(energies)
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_energies, np.log(fluxes))
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value
        }

def analyze_standard_oscillation(energies, fluxes, baseline=100):
    """
    Analyze standard neutrino oscillation behavior
    
    Parameters:
    energies (array): Neutrino energies
    fluxes (array): Measured fluxes
    baseline (float): Baseline distance (km)
    
    Returns:
    dict: Analysis results
    """
    # Create analyzer instance
    analyzer = NeutrinoAnalyzer()
    
    # Fit oscillation model
    fit_results = analyzer.fit_oscillation_model(energies, fluxes, baseline)
    
    # Calculate expected oscillations
    expected_fluxes = []
    for E in energies:
        # Standard 3-neutrino mixing
        prob = analyzer.calculate_oscillation_probability(E, baseline, 7.5e-5, np.pi/4)
        expected_fluxes.append(prob)
    
    return {
        'fit_results': fit_results,
        'expected_fluxes': expected_fluxes,
        'data_fluxes': fluxes.tolist()
    }

def check_sterile_neutrino(energies, fluxes, baseline=100, sterile_params=None):
    """
    Check for sterile neutrino signatures
    
    Parameters:
    energies (array): Neutrino energies
    fluxes (array): Measured fluxes
    baseline (float): Baseline distance (km)
    sterile_params (dict): Parameters for sterile neutrino model
    
    Returns:
    dict: Sterile neutrino analysis results
    """
    if sterile_params is None:
        sterile_params = {'delta_m2': 0.01, 'theta': 0.1}
    
    analyzer = NeutrinoAnalyzer()
    
    # Calculate standard oscillation probability
    standard_probs = []
    for E in energies:
        prob = analyzer.calculate_oscillation_probability(E, baseline, 7.5e-5, np.pi/4)
        standard_probs.append(prob)
    
    # Calculate sterile oscillation probability
    sterile_probs = []
    for E in energies:
        prob = analyzer.calculate_oscillation_probability(E, baseline, 
                                                         sterile_params['delta_m2'], 
                                                         sterile_params['theta'])
        sterile_probs.append(prob)
    
    # Compare with data
    chi2_standard = np.sum((np.array(fluxes) - np.array(standard_probs))**2)
    chi2_sterile = np.sum((np.array(fluxes) - np.array(sterile_probs))**2)
    
    return {
        'chi2_standard': chi2_standard,
        'chi2_sterile': chi2_sterile,
        'likelihood_ratio': chi2_standard - chi2_sterile,
        'sterile_params': sterile_params
    }

def bayesian_analysis(energies, fluxes, baseline=100):
    """
    Perform Bayesian analysis of neutrino data
    
    Parameters:
    energies (array): Neutrino energies
    fluxes (array): Measured fluxes
    baseline (float): Baseline distance (km)
    
    Returns:
    dict: Bayesian analysis results
    """
    # Define Bayesian model using PyMC3
    with pm.Model() as model:
        # Priors
        delta_m2 = pm.Normal('delta_m2', mu=7.5e-5, sigma=1e-6)
        theta = pm.Uniform('theta', lower=0, upper=np.pi/2)
        
        # Likelihood
        @pm.deterministic
        def oscillation_prob(delta_m2=delta_m2, theta=theta):
            probs = []
            for E in energies:
                prob = 0.5 * np.sin(2*theta)**2 * np.sin(1.27 * delta_m2 * baseline / E)**2
                probs.append(prob)
            return np.array(probs)
        
        # Observed data
        observed = pm.Normal('fluxes', mu=oscillation_prob, sigma=0.1, observed=fluxes)
        
        # Sample
        trace = pm.sample(1000, tune=1000, chains=2, cores=1, return_inferencedata=False)
    
    # Extract posterior statistics
    posterior_stats = {
        'delta_m2_mean': np.mean(trace['delta_m2']),
        'delta_m2_std': np.std(trace['delta_m2']),
        'theta_mean': np.mean(trace['theta']),
        'theta_std': np.std(trace['theta'])
    }
    
    return {
        'posterior_stats': posterior_stats,
        'trace': trace
    }

def plot_neutrino_flux(energies, fluxes, title="Neutrino Flux vs Energy"):
    """
    Plot neutrino flux measurements
    
    Parameters:
    energies (array): Neutrino energies
    fluxes (array): Measured fluxes
    title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(energies, fluxes, 'bo-', label='Measured Flux')
    plt.xlabel('Energy (GeV)')
    plt.ylabel('Flux')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_model_comparison(energies, fluxes, standard_probs, sterile_probs, title="Model Comparison"):
    """
    Plot comparison between standard and sterile neutrino models
    
    Parameters:
    energies (array): Neutrino energies
    fluxes (array): Measured fluxes
    standard_probs (array): Standard model probabilities
    sterile_probs (array): Sterile model probabilities
    title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(energies, fluxes, 'ko-', label='Measured Flux')
    plt.plot(energies, standard_probs, 'b--', label='Standard Model')
    plt.plot(energies, sterile_probs, 'r:', label='Sterile Model')
    plt.xlabel('Energy (GeV)')
    plt.ylabel('Probability')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()