## proof of central limit theorem with monte carlo simulation
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

## create underlying non-normal distribution (exponential distribution with scale parameter 1)
np.random.seed(42)
pop = np.random.exponential(scale=1, size=100000)

## define function to compute sample means
def simulate_clt(n_samples, n_obs):
    sample_means = []
    for i in range(n_samples):
        sample = np.random.choice(pop, size=(n_obs), replace=True) ## create a sample of n_obs observations from the population, with replacement
        means = np.mean(sample)
        sample_means.append(means)
    return np.array(sample_means)

## call three times, storing each result, print n_obs value, mean and std of sample means, and skewness of sample means
results = {}
obs_nums = [5, 30, 100]
for n_obs in obs_nums:
    sample_means = simulate_clt(1000, n_obs)
    mean = np.mean(sample_means)
    std = np.std(sample_means)
    skew = stats.skew(sample_means)
    results[n_obs] = (mean, std, skew)  
    print(f"n_obs: {n_obs}, mean of sample means: {np.mean(sample_means):.4f}, std of sample means: {np.std(sample_means):.4f}, skewness of sample means: {stats.skew(sample_means):.4f}") ## need to store these values right?

## create one histogram with three distributions: the three sets of sample means
plt.figure(figsize=(12, 8))
for n_obs in obs_nums:
    sample_means = simulate_clt(1000, n_obs)
    plt.hist(sample_means, bins=50, alpha=0.5, label=f'Sample Means (n={n_obs})', density=True)
plt.title("Central Limit Theorem Demonstration")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig('clt_demo.png')
plt.show()