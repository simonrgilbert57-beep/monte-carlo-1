
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

## Define the target distribution as an unnormalized mixture of two Gaussians using scipy.states.norm.pdf
def target_distribution(x):
    return 0.4 * norm.pdf(x, loc=-2, scale=0.8) + 0.6 * norm.pdf(x, loc=3, scale=1.2) ## Defines target distribution. Loc = location of the mean, scale = standard deviation. Mixture of two Gaussians with weights 0.4 and 0.6
## define metropolis-hastings algorithm
def metropolis_hastings(target_dist, n_samples, x_init):
    samples = [] ## initialize list to hold samples
    x_current = x_init ## initialize current sample to initial value (input of the function)
    n_accepted = 0 ## initialize counter for accepted proposals
    for i in range(n_samples):
        mu_proposal = np.log(abs(x_current) + 1) ## propose new sample from normal distribution centered at log of absolute value of current sample plus 1
        x_proposal = np.random.normal(loc=mu_proposal, scale=0.5) ## propose new sample from normal distribution centered at mu_proposal with std of 0.5
        mu_reverse = np.log(abs(x_proposal) + 1) ## compute reverse proposal mean as log of absolute value of current sample plus 1  
        q_forward = norm.pdf(x_proposal, loc=mu_proposal, scale=0.5) ## compute forward proposal density
        q_backward = norm.pdf(x_current, loc=mu_reverse, scale=0.5) ## compute backward proposal density
        alpha = (target_dist(x_proposal) * q_backward) / (target_dist(x_current) * q_forward) ## define acceptance ratio as (target density at proposed sample * backward proposal density) / (target density at current sample * forward proposal density) according to metropolis-hastings acceptance ratio formula
        alpha_1 = min(1, alpha) ## acceptance probability is minimum of 1 and acceptance ratio to bound acceptance probability between 0 and 1
        if np.random.rand() < alpha_1:## accept proposal if random number between 0 and 1 is less than acceptance ratio
            x_current = x_proposal
            n_accepted += 1
        samples.append(x_current)
    return np.array(samples), n_accepted

## run metropolis-hastings algorithm to generate samples from target distribution
n_samples_0 = 10000 
x_init_0 = 0
samples, n_accepted = metropolis_hastings(target_distribution, n_samples_0, x_init_0)
## add burn-in period by discarding the first 1000 samples
burn_in = 1000
samples = samples[burn_in:]

## plot 1: trace plot — chain values over iterations (post burn-in), x-axis = iteration, y-axis = sampled value
plt.figure(figsize=(12, 6))
plt.plot(samples, alpha=0.5)
plt.title("Trace Plot of Metropolis-Hastings Samples")
plt.xlabel("Iteration")
plt.ylabel("Sampled Value")
plt.tight_layout()
plt.savefig('mh_trace_plot.png')   

## plot 2: histogram of post-burn-in samples overlaid with the true target density
x = np.linspace(-6, 8, 500)
plt.figure(figsize=(12, 6))
plt.hist(samples, bins=60, density=True, color='g', label='MH Samples') 
plt.plot(x, target_distribution(x), color='r', label='True Target Density', linewidth=2)
plt.title("Histogram of Metropolis-Hastings Samples with True Target Density")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig('mh_histogram.png')

## print acceptance rate, mean, std of post-burn-in samples
acceptance_rate = n_accepted / n_samples_0 ## compute acceptance rate as number of accepted proposal divided by total number of proposals
print(f"Acceptance Rate: {acceptance_rate:.2%}")
print(f"Mean of Samples: {np.mean(samples):.4f}")
print(f"Standard Deviation of Samples: {np.std(samples):.4f}")  