import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(41)

## define function that takes a distribution name as a string and a numpy array of samples, and prints The distribution name, Sample mean, standard deviation, and median — all rounded to 4 decimal places
def sample_and_sum(dist_name, samples):
    mean = np.mean(samples)
    std = np.std(samples)
    median = np.median(samples)
    print(f"{dist_name} - Sample mean: {mean:.4f}, Sample std: {std:.4f}, Sample median: {median:.4f}")

## Draw three sets of samples, Normal, Uniform, and Exponential, each with 10000 samples, and call the function on each set of samples
normal = np.random.normal(0, 1, 10000)
uniform = np.random.uniform(0, 1, 10000)
exponential = np.random.exponential(1, 10000)   
dist_types = ["normal", "uniform", "exponential"]
dist_arrays = [normal, uniform, exponential]
for name, array in zip(dist_types, dist_arrays):
    sample_and_sum(name, array)

## true mean definition for each distribution type
true_means = {
    "normal": 0,
    "uniform": 0.5,
    "exponential": 1
}

## sample size for the demo
sample_size_1 = [10, 100, 1000, 10000, 100000]

## empiricially prove Law of Large Numbers for each distribution type
def lln_demo(dist_type, true_mean, sample_size):
    sample_means = []
    for i in sample_size:
        if dist_type == "normal":
            samples = np.random.normal(0, 1, i)
        elif dist_type == "uniform":
            samples = np.random.uniform(0, 1, i)
        elif dist_type == "exponential":
            samples = np.random.exponential(1, i)
        sample_means.append(np.mean(samples)-true_mean)
    return sample_means

## compute
plt.figure(figsize=(12, 8))
for i in dist_types:
    errors = lln_demo(i, true_means[i], sample_size_1)
    plt.plot(sample_size_1, errors, label=i)
plt.xscale('log')
plt.axhline(0, color='black', linestyle='dashed', label='Zero error')
plt.title("Law of Large Numbers Demonstration")
plt.xlabel("Sample Size (log scale)")
plt.ylabel("Sample Mean")  
plt.legend()
plt.tight_layout()
plt.savefig('lln_demo.png') 

## five stacked histograms of the three distributions, with sample sizes of 10, 100, 1000, 10000, and 100000, with the x-axis on a log scale 
fig, axes = plt.subplots(5, 1, figsize=(12, 20))
for idx, n in enumerate(sample_size_1):
    ax = axes[idx]
    for dist, color in zip(dist_types, ['blue', 'orange', 'green']):
        if dist == "normal":
            samples = np.random.normal(0, 1, n)
        elif dist == "uniform":
            samples = np.random.uniform(0, 1, n)
        elif dist == "exponential":
            samples = np.random.exponential(1, n)
        ax.hist(samples, bins=50, alpha=0.5, label=dist, color=color, density=True)
        ax.axvline(true_means[dist], color=color, linestyle='dashed', label=f'{dist} true mean', linewidth=2)
    ax.set_title(f'Sample Size: {n}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
plt.tight_layout()
plt.savefig('lln_histograms.png')
