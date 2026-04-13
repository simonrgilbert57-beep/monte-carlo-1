import numpy as np
import matplotlib.pyplot as plt

## define function monte carlo price
def sim_price_paths(s0, mu, sigma, t, dt, n_sims):
    arr = []
    for i in range(n_sims):
        s = s0
        prices = []
        for j in range(int(t/dt)):
            s = s * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.standard_normal())
            prices.append(s)
        arr.append(prices)
    return np.array(arr)

## return 2d numpy array of simulated price paths and results
arr = sim_price_paths(100, 0.07, 0.2, 1, 1/252, 1000)
print(arr.shape) ## should be (252, 10000)
print(np.mean(arr[:,-1])) ## print mean of final prices
print(np.percentile(arr, 95))
print(np.percentile(arr, 5))

## graphing

## histogram of final prices with 50 bins
plt.hist(arr[:,-1], bins=50)
plt.title("Distribution of Simulated Final Prices")
plt.xlabel("Terminal Price")
plt.ylabel("Frequency")

## add vertical line for 5th percentile and 95th percentile
plt.axvline(np.percentile(arr[:,-1], 5), color='red', linestyle='dashed', label='5th Percentile')
plt.axvline(np.percentile(arr[:,-1], 95 ), color='green', linestyle='dashed', label='95th Percentile')    
plt.legend()
plt.tight_layout()
plt.savefig('monte_carlo_dist.png')
plt.show()