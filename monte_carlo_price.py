import numpy as np
import matplotlib.pyplot as plt

## define function monte carlo price
def sim_price_paths(s0, mu, sigma, t, dt, n_sims): ## s0 = initial price, mu = expected return, sigma = volatility, t = time horizon, dt = time step, n_sims = number of simulations
    arr = [] ## defines array to hold price paths
    for i in range(n_sims): ## you can loop over a number? not just a list? 
        s = s0 # resets price to initial price for each simulation
        prices = [] ## resets price path for each simulation
        for j in range(int(t/dt)):
            s = s * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.standard_normal()) 
            prices.append(s)
        arr.append(prices)
    return np.array(arr)
