import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import norm

## define priors
mu0 = 0
sig0 = 1
sig = 1

## Generate a dataset of 100 observations drawn from a normal distribution with true mean 2.0 and std 1
sample_1 = np.random.normal(2.0, 1, 100)

## define function to iterate through the data one observation at a time, updating the posterior mean and variance after each observation, and returns two lists: the posterior means and posterior stds at each step
def bayes_update(mu0, sig0, sig, data):
    post_means = []
    post_stds = []
    for x in data:
        post_var = 1 / (1/sig0**2 + 1/sig**2) ## compute posterior variance
        post_mean = post_var * (mu0/sig0**2 + x/sig**2) ## compute posterior mean
        post_means.append(post_mean)
        post_stds.append(np.sqrt(post_var))
        mu0 = post_mean ## update prior mean to posterior mean for next iteration
        sig0 = np.sqrt(post_var) ## update prior std to posterior std for next iteration
    return post_means, post_stds

## call function on sample_1 and store results
post_means_1, post_stds_1 = bayes_update(mu0, sig0, sig, sample_1)

## plot the evolution of posterior mean and a shaded +/- 1 std region around the mean as more data is observed
plt.figure(figsize=(12, 6))
plt.plot(post_means_1, label='Posterior Mean')
plt.fill_between(range(len(post_means_1)), np.array(post_means_1) - np.array(post_stds_1), np.array(post_means_1) + np.array(post_stds_1), color='blue', alpha=0.2, label='Posterior Std Dev')
plt.axhline(2.0, color='red', linestyle='dashed', label='True Mean')
plt.axhline(0.0, color='gray', linestyle='dashed', label='Prior Mean')
plt.title("Bayesian Updating of Mean with Normal Likelihood and Normal Prior")
plt.xlabel("Number of Observations")
plt.ylabel("Mean Estimate") 
plt.legend()
plt.tight_layout()
plt.savefig('bayes_update_demo.png')

## distributions changing over time plot
snapshot_intervals = [1, 2, 5, 10, 20, 40, 60, 80, 90, 100]
x_range = np.linspace(-1, 4, 500)
colors = plt.cm.Blues(np.linspace(0.2, 1.0, len(snapshot_intervals)))
plt.figure(figsize=(12,6))
for n, color in zip(snapshot_intervals, colors):
    mu = post_means_1[n-1]
    std = post_stds_1[n-1]
    plt.plot(x_range, norm.pdf(x_range, mu, std), color=color, 
             label=f'n={n}' if n in [1, 10, 100] else None)
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Evolution of Posterior Distributions")
plt.legend()
plt.tight_layout()
plt.savefig('posterior_distributions.png')  