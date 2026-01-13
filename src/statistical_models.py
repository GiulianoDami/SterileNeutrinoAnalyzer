import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import emcee
import pymc3 as pm

def bayesian_inference(data, model):
    """
    Perform Bayesian inference on neutrino oscillation data using the specified model.
    
    Parameters:
    data (array-like): Neutrino oscillation measurements
    model (object): Statistical model to use for inference
    
    Returns:
    dict: Dictionary containing posterior samples and summary statistics
    """
    # Convert data to numpy array
    data = np.array(data)
    
    # Define the Bayesian model using PyMC3
    with pm.Model() as bayesian_model:
        # Priors for model parameters
        if hasattr(model, 'prior'):
            # Use provided prior if available
            pass
        else:
            # Default uniform priors
            mu = pm.Normal('mu', mu=0, sigma=10)
            sigma = pm.HalfNormal('sigma', sigma=5)
            
        # Likelihood
        likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=data)
        
        # Sample from posterior
        trace = pm.sample(1000, tune=1000, chains=2, return_inferencedata=True)
        
    # Extract posterior samples
    posterior_samples = {}
    for var in trace.posterior.var():
        posterior_samples[var] = trace.posterior[var].values.flatten()
    
    # Calculate summary statistics
    summary_stats = {
        'mean': {var: np.mean(posterior_samples[var]) for var in posterior_samples},
        'std': {var: np.std(posterior_samples[var]) for var in posterior_samples},
        'median': {var: np.median(posterior_samples[var]) for var in posterior_samples}
    }
    
    return {
        'posterior_samples': posterior_samples,
        'summary_statistics': summary_stats,
        'trace': trace
    }

def compare_models(model1, model2):
    """
    Compare two statistical models using Bayes factors.
    
    Parameters:
    model1 (object): First statistical model
    model2 (object): Second statistical model
    
    Returns:
    dict: Dictionary containing model comparison results
    """
    # For simplicity, we'll implement a basic comparison based on log evidence
    # In practice, this would involve more sophisticated methods like thermodynamic integration
    
    # Simulate log evidence calculation (this is a placeholder implementation)
    log_evidence_1 = -np.random.exponential(1)  # Random log evidence for model 1
    log_evidence_2 = -np.random.exponential(1)  # Random log evidence for model 2
    
    # Bayes factor
    bayes_factor = np.exp(log_evidence_1 - log_evidence_2)
    
    # Model weights (normalized)
    weight_1 = np.exp(log_evidence_1) / (np.exp(log_evidence_1) + np.exp(log_evidence_2))
    weight_2 = np.exp(log_evidence_2) / (np.exp(log_evidence_1) + np.exp(log_evidence_2))
    
    return {
        'bayes_factor': bayes_factor,
        'model_weights': {'model1': weight_1, 'model2': weight_2},
        'log_evidence': {'model1': log_evidence_1, 'model2': log_evidence_2}
    }

def calculate_confidence_level(results):
    """
    Calculate confidence level from Bayesian inference results.
    
    Parameters:
    results (dict): Results from bayesian_inference function
    
    Returns:
    float: Confidence level (0-1)
    """
    # Extract posterior samples
    posterior_samples = results['posterior_samples']
    
    # Calculate credible intervals (95% by default)
    credible_intervals = {}
    for var, samples in posterior_samples.items():
        credible_intervals[var] = np.percentile(samples, [2.5, 97.5])
    
    # Calculate probability that parameter is positive (example metric)
    prob_positive = {}
    for var, samples in posterior_samples.items():
        prob_positive[var] = np.mean(samples > 0)
    
    # Simple confidence level based on how much of the distribution is in a meaningful range
    # This is a simplified example - in practice, you'd want to define what constitutes "significant"
    mean_values = results['summary_statistics']['mean']
    std_values = results['summary_statistics']['std']
    
    # Calculate z-scores for each parameter
    z_scores = {var: abs(mean_values[var]) / std_values[var] if std_values[var] != 0 else 0 
                for var in mean_values}
    
    # Convert z-scores to confidence levels (using standard normal CDF)
    confidence_levels = {var: stats.norm.cdf(z_score) * 2 - 1 for var, z_score in z_scores.items()}
    
    # Return average confidence level across all parameters
    avg_confidence = np.mean(list(confidence_levels.values()))
    
    return max(0, min(1, avg_confidence))  # Ensure between 0 and 1