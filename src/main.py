import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import emcee
import pymc3 as pm
import os

def main():
    # Demonstrate basic functionality of the analyzer
    print("Sterile Neutrino Analyzer")
    print("=" * 30)
    
    # Generate sample data
    np.random.seed(42)
    n_events = 1000
    true_flux = 50.0
    background = 10.0
    
    # Simulate observed data with some noise
    observed_data = np.random.poisson(true_flux + background, n_events)
    
    # Basic statistics
    mean_observed = np.mean(observed_data)
    std_observed = np.std(observed_data)
    
    print(f"Mean observed events: {mean_observed:.2f}")
    print(f"Standard deviation: {std_observed:.2f}")
    
    # Bayesian analysis using PyMC3
    with pm.Model() as model:
        # Prior for the true flux
        flux = pm.Normal('flux', mu=50, sigma=10)
        
        # Likelihood
        likelihood = pm.Poisson('likelihood', mu=flux, observed=observed_data)
        
        # Sample from posterior
        trace = pm.sample(1000, tune=1000, chains=2, return_inferencedata=True)
    
    # Extract posterior samples
    posterior_samples = trace.posterior['flux'].values.flatten()
    
    # Calculate credible intervals
    lower_ci = np.percentile(posterior_samples, 2.5)
    upper_ci = np.percentile(posterior_samples, 97.5)
    
    print(f"95% Credible interval for flux: [{lower_ci:.2f}, {upper_ci:.2f}]")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of observed data
    ax1.hist(observed_data, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(mean_observed, color='red', linestyle='--', 
               label=f'Mean: {mean_observed:.2f}')
    ax1.set_xlabel('Observed Events')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Observed Events')
    ax1.legend()
    
    # Posterior distribution
    ax2.hist(posterior_samples, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(np.mean(posterior_samples), color='red', linestyle='--',
               label=f'Mean: {np.mean(posterior_samples):.2f}')
    ax2.axvline(true_flux, color='black', linestyle='-', 
               label=f'True Flux: {true_flux}')
    ax2.set_xlabel('Flux (events)')
    ax2.set_ylabel('Density')
    ax2.set_title('Posterior Distribution of Flux')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('results.png')
    print("Results saved to results.png")

if __name__ == "__main__":
    main()