import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

## define DCF function
def dcf(cash_flow, growth_r, discount_r, t):
    for i in range(t):
        cash_flow = cash_flow * (1 + growth_r) / (1 + discount_r)
    return cash_flow

## define base case parameters
cash_flow = 100
t = 10

## create two arrays of growth rates and discount rates
growth_rates = np.array([])
discount_rates = np.array([])
t_values = 5
for i in range(t_values):
    growth_rates = np.append(growth_rates, 0.01 + i*0.01) ## creates array of growth rates from 1% to 5% in increments of 1%
    discount_rates = np.append(discount_rates, 0.06 + i*0.01) ## creates array of discount rates from 5% to 9% in increments of 1%

## Build a 5x5 numpy array where each cell is the DCF value for the corresponding growth rate and discount rate
dcf_values = np.zeros((t_values, t_values))
for i in range(t_values):
    for j in range(t_values):
        dcf_values[i,j] = dcf(cash_flow, growth_rates[i], discount_rates[j], t)

## Convert the result to a pandas DataFrame with growth_rates as the index and discount_rates as the columns. Round all values to 2 decimal places
df = pd.DataFrame(dcf_values, index=growth_rates, columns=discount_rates).round(2)
print(df)


## create heatmap of DCF values with growth rates on the y-axis and discount rates on the x-axis, annotating each cell with the DCF value at one decimal place, red for low value and green for high value, and 0.5 width cell borders 
def plot_heatmap(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt=".1f", cmap="RdYlGn", linewidths=0.5)
    plt.title("DCF Values for Different Growth Rates and Discount Rates")   
    plt.xlabel("Discount Rate")
    plt.ylabel("Growth Rate")
    plt.tight_layout()
    plt.savefig('dcf_heatmap.png')
    plt.show()
    return
plot_heatmap(df)