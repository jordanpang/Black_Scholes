"""
Module name: REPL_Application

This module provides a REPL (Read-Eval-Print Loop) application using the Black-Scholes Theoretical Pricing Model.
It is designed as a starting point to showcase the calculation of European options pricing.

Main functions: black_scholes
"""

import numpy as np
from scipy.stats import norm

def black_scholes(r, X, S, t, sigma, option_type):
    """
    Calculate the theoretical value of a call or put option using the Black-Scholes model.

    Parameters:
        r (float): The annual risk-free interest rate (in decimal form, e.g., 0.05 for 5%)
        X (float): The strike price of the option
        S (float): The current price of the underlying asset (non-dividend-paying stock)
        t (float): The time to expiration, in years
        sigma (float): The annualized volatility of the underlying asset (in decimal form, e.g. 0.2 for 20%)
        option_type (str): The type of option ('c' for call, 'p' for put)
    
    Returns:
        float: The theoretical price of the option
    """
    try:
        # Ensure inputs are floats
        r = float(r)
        X = float(X)
        S = float(S)
        t = float(t)
        sigma = float(sigma)
        
        # Check that X and S are non-negative
        if X < 0 or S < 0:
            raise ValueError("Strike price (X) and underlying price (S) must be non-negative.")
        # Calculate d1 and d2 equations
        d1 = (np.log(S/X) + (r + sigma**2 / 2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)

        # Calculate option price based on type
        if option_type == 'c':
            price = S * norm.cdf(d1) - X * np.exp(-r * t) * norm.cdf(d2)
        elif option_type == 'p':
            price = X * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("Invalid option type. Please use 'c' for call or 'p' for put.")
        
        return price
    
    except ValueError as err:
        print(f"Error: {err}")
        return None

def main():
    while True:
        user_input = input("Type 'y' to continue, or 'n' to exit: ").strip().lower()
        if user_input == 'y':
            try:
                r = float(input("Enter interest rate (e.g., 0.05 for 5%): "))
                X = float(input("Enter strike price: "))
                S = float(input("Enter underlying price: "))
                t = float(input("Enter time to expiry, in years: "))
                sigma = float(input("Enter annual volatility (e.g., 0.2 for 20%): "))
                option_type = input("Enter 'c' for a call option or 'p' for a put option: ").strip().lower()

                price = black_scholes(r, X, S, t, sigma, option_type)
                if price is not None:
                    print(f"The theoretical price of the option is: {round(price, 3)}")
            
            except ValueError:
                print("Invalid input. Please ensure all numerical inputs are valid numbers.")
        
        elif user_input == 'n':
            break
        else:
            print("Invalid input. Please type 'y' to continue or 'n' to exit.")

if __name__ == "__main__":
    main()