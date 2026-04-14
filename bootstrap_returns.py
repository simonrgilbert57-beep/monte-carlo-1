## reate a numpy array of 252 synthetic daily log returns drawn from a normal distribution
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42) 
arr = np.random.normal(0.0003, 0.012, 252)

## Define a function to compute bootstrap returns
def bootstrap_stat(returns, n_boot, stat_func):
    bootstrap_stats = []
    for i in range(n_boot):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        stat = stat_func(sample)
        bootstrap_stats.append(stat)
    return np.array(bootstrap_stats)

## Call the function twice — once with np.mean, once with np.std
bootstrap_means = bootstrap_stat(arr, 10000, np.mean)
bootstrap_stds = bootstrap_stat(arr, 10000, np.std)
prints = [bootstrap_means, bootstrap_stds]
names = ["Mean", "Std"]
for label, x in zip(names, prints):
    print(f" {label} Bootstrap mean: {np.mean(x):.6f}, {label} Bootstrap std: {np.std(x):.6f}")
    print(f" {label} 2.5% percentile: {np.percentile(x, 2.5):.6f}, {label} 97.5% percentile: {np.percentile(x, 97.5):.6f}")

