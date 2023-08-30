import numpy as np
from scipy.stats import norm
from py_vollib.black_scholes import black_scholes as bs

# Define variables 
r = 0.01 #Interest rate
S = 30 # Underlying price
K = 40 #Strike price
T = 240/365 #Time to expiration in years
sigma = 0.30 #Volatility

def blackScholes(r, S, K, T, sigma, type="c"):
    "Calculate BS price of call/put"
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    try:
        if type == "c":
            price = S*norm.cdf(d1, 0, 1) - K*np.exp(-r*T)*norm.cdf(d2, 0, 1)
        elif type == "p":
            price = K*np.exp(-r*T)*norm.cdf(-d2, 0, 1) - S*norm.cdf(-d1, 0, 1)
        return price, bs(type, S, K, T, r, sigma)
    except:
        print("Please confirm option type, either 'c' for Call or 'p' for Put!")


option_type = 'c'
print("Option Price: ", [round(x,3) for x in blackScholes(r, S, K, T, sigma, option_type)])