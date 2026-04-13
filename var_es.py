## import monte carlo price function
from monte_carlo_price import sim_price_paths 
import numpy as np


## run simulation again
arr = sim_price_paths(100, 0.07, 0.2, 1, 1/252, 1000)
arr_f = arr[:,-1] ## get final prices from simulation
##compute returns for each simulation
arr_returns = (arr_f - 100) / 100 ## compute returns as (final price - initial price) / initial price

## write function to compute VaR and ES
def compute_var_es(returns, confidence_level):
    var = np.percentile(returns, (1 - confidence_level)*100) ## compute VaR at given confidence level
    es = np.mean([r for r in returns if r < var]) ## compute ES as mean of returns below VaR
    print(f"VaR at {confidence_level*100}% confidence level: {-var:.2%}")
    print(f"ES at {confidence_level*100}% confidence level: {-es:.2%}")
    return var, es

## compute Var and ES at 95% confidence level
compute_var_es(arr_returns, 0.95)
