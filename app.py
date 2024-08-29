import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Set the page configuration
st.set_page_config(
    page_title="Custom Streamlit App",
    page_icon="📈",
    layout="wide",
)


# First make the Black-Scholes function


class BlackScholes:
    """
    Calculate the theoretical value of a call or put using the BS model.
    Add the risk measures to build layers.

    Parameters:
        r (float): The annual risk-free interest rate
                   (in decimal form, e.g., 0.05 for 5%)
        X (float): The strike price of the option
        S (float): The current price of the underlying asset
                   (non-dividend-paying stock)
        t (float): The time to expiration, in years
        sigma (float): The annualized volatility of the underlying asset
                       (in decimal form, e.g. 0.2 for 20%)

    Returns:
        float: The theoretical price of the option
    """

    def __init__(self, r, X, S, t, sigma):
        self.r = r
        self.X = X
        self.S = S
        self.t = t
        self.sigma = sigma

    def compute(self):
        # Ensure inputs are floats
        r = float(self.r)
        X = float(self.X)
        S = float(self.S)
        t = float(self.t)
        sigma = float(self.sigma)

        # Calculate d1 and d2 equations
        d1 = (np.log(S / X) + (r + sigma**2 / 2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)

        # Calculate option price
        call_price = S * norm.cdf(d1) - X * np.exp(-r * t) * norm.cdf(d2)
        put_price = X * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return call_price, put_price

    def delta(self):
        # Ensure inputs are floats
        r = float(self.r)
        X = float(self.X)
        S = float(self.S)
        t = float(self.t)
        sigma = float(self.sigma)

        # Calculate d1 and d2 equations
        d1 = (np.log(S / X) + (r + sigma**2 / 2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)

        # Calculate the deltas
        call_delta = norm.cdf(d1, 0, 1)
        put_delta = -norm.cdf(-d1, 0, 1)

        return call_delta, put_delta

    def gamma(self):
        # Ensure inputs are floats
        r = float(self.r)
        X = float(self.X)
        S = float(self.S)
        t = float(self.t)
        sigma = float(self.sigma)

        # Calculate d1 equation
        d1 = (np.log(S / X) + (r + sigma**2 / 2) * t) / (sigma * np.sqrt(t))

        gamma_calc = norm.pdf(d1, 0, 1) / (S * sigma * np.sqrt(t))

        return gamma_calc, gamma_calc

    def vega(self):
        # Ensure inputs are floats
        r = float(self.r)
        X = float(self.X)
        S = float(self.S)
        t = float(self.t)
        sigma = float(self.sigma)

        # Calculate d1 equation
        d1 = (np.log(S / X) + (r + sigma**2 / 2) * t) / (sigma * np.sqrt(t))

        vega_calc = S * norm.pdf(d1, 0, 1) * np.sqrt(t)

        return vega_calc * 0.01, vega_calc * 0.01

    def theta(self):
        # Ensure inputs are floats
        r = float(self.r)
        X = float(self.X)
        S = float(self.S)
        t = float(self.t)
        sigma = float(self.sigma)

        # Calculate d1 and d2 equations
        d1 = (np.log(S / X) + (r + sigma**2 / 2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)

        call_theta = -S * norm.pdf(d1, 0, 1) * sigma / (
            2 * np.sqrt(t)
        ) - r * X * np.exp(-r * t) * norm.cdf(d2, 0, 1)
        put_theta = -S * norm.pdf(d1, 0, 1) * sigma / (2 * np.sqrt(t)) + r * X * np.exp(
            -r * t
        ) * norm.cdf(-d2, 0, 1)

        return call_theta / 365, put_theta / 365

    def rho(self):
        # Ensure inputs are floats
        r = float(self.r)
        X = float(self.X)
        S = float(self.S)
        t = float(self.t)
        sigma = float(self.sigma)

        # Calculate d1 and d2 equations
        d1 = (np.log(S / X) + (r + sigma**2 / 2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)

        call_rho = X * t * np.exp(-r * t) * norm.cdf(d2, 0, 1)
        put_rho = -X * t * np.exp(-r * t) * norm.cdf(-d2, 0, 1)

        return call_rho * 0.01, put_rho * 0.01


def main():
    st.title("The Black-Scholes Theoretical Pricing Model")

    st.subheader("Developed by Jordan Pang")
    # Contact details
    st.markdown(
        """
    **Contact Information**:
    - **Email**: jordanpang03@outlook.com
    - **GitHub**: [jordanpang](https://github.com/jordanpang/)
    """
    )

    # Define the tabs
    tab1, tab2, tab3, tab4 = st.tabs(["About", "Option Pricer", "Heatmap", "Data"])

    # Home Tab
    with tab1:
        st.write(
            """
            This project presents an interactive application, that I developed using
            Streamlit, an open-source framework, to explore and understand how the
            Black-Scholes Option Pricing Model can be applied in programming for both
            theoretical insights and practical demonstrations. In particular, this is
            intended to allow users to input factors, to calculate theoretical values
            of options dynamically, providing visual aids by means of heatmaps,
            to illustrate the impact of sensitive inputs to option prices. As a result,
            P&L for taking option positions are calculated and a heatmap is produced to distinctly
            portraying how sensitive the position is to changes in volatility and underlying price.
            The next goal is to store input values for the Black-Scholes options pricing model,
            and then to create an output table recording the volatility and underlying shocks, and the
            P&L associated with the postion, where each record is uniquely identified and connected by a
            Calculation ID.

            The motivation for the project stems from Natenburg's 'Option Volatility and Pricing',
            which gave me insights into various methods to price an option, one of which is
            the Black-Scholes Model for European options. I began by creating a python REPL
            application, taking in the five inputs to an option's price and then outputs a
            call or put value. To extend this, I thought it would be interesting to add
            a GUI layer, through Streamlit, to create a more interactive environment to display
            my ideas distinctly and clearly. Over time, I have built more layers to this project,
            which leads to the current state of this page today.
        """
        )

    # Data Tab
    with tab2:
        st.header("Option Pricer")
        st.write(
            "Please adjust the follow inputs to compute your theoretical value of options."
        )

        S = st.number_input("Underlying Price", value=100.0)
        X = st.number_input("Strike Price", value=100.0)
        t = st.number_input("Time to Expiration (Years)", value=1.0)
        sigma = st.number_input("Volatility", value=0.2)
        r = st.number_input("Interest Rate", value=0.05)

        black_scholes_model = BlackScholes(r, X, S, t, sigma)
        call_value, put_value = black_scholes_model.compute()

        st.write("---")
        st.subheader("Result: ")

        # Define custom CSS for styling
        css = """
        <style>
        .metric-box {
            border-radius: 10px;
            padding: 10px;
            color: white;
            font-size: 24px;
            text-align: center;
            font-weight: bold;
            display: inline-block;
            margin: 10px;
        }
        .call-box {
            background-color: #0000FF; /* Blue for Call */
        }
        .put-box {
            background-color: #ee7600; /* Orange for Put */
        }
        .risk-box {
            background-color: #AA336A; /* Pink for Risk Measures */
        }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

        # Display Call and Put Values in colored tables
        col1, col2 = st.columns([1, 0.5])  # Adjust ratio spacing
        with col1:
            st.markdown(
                f"""
            <div class="metric-box call-box">
                Call Value: ${round(call_value, 2)}
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
            <div class="metric-box put-box">
                Put Value: ${round(put_value, 2)}
            </div>
            """,
                unsafe_allow_html=True,
            )
        st.write("")
        st.write("")
        st.write("")

        # Compute the risk measures
        call_delta, put_delta = black_scholes_model.delta()
        call_gamma, put_gamma = black_scholes_model.gamma()
        call_vega, put_vega = black_scholes_model.vega()
        call_theta, put_theta = black_scholes_model.theta()
        call_rho, put_rho = black_scholes_model.rho()

        st.markdown(
            "Details on the Greek risk measures associated to each option are given below: "
        )

        col3, col4 = st.columns([1, 0.5], gap="small")

        with col3:
            st.markdown(
                f"""
            <div class="metric-box risk-box">
                Call Delta: {round(call_delta, 3)}
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
            <div class="metric-box risk-box">
                Call Gamma: {round(call_gamma, 3)}
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
            <div class="metric-box risk-box">
                Call Vega: {round(call_vega, 3)}
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
            <div class="metric-box risk-box">
                Call Theta: {round(call_theta, 3)}
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
            <div class="metric-box risk-box">
                Call Rho: {round(call_rho, 3)}
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col4:
            st.markdown(
                f"""
            <div class="metric-box risk-box">
                Put Delta: {round(put_delta, 3)}
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
            <div class="metric-box risk-box">
                Put Gamma: {round(put_gamma, 3)}
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
            <div class="metric-box risk-box">
                Put Vega: {round(put_vega, 3)}
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
            <div class="metric-box risk-box">
                Put Theta: {round(put_theta, 3)}
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
            <div class="metric-box risk-box">
                Put Rho: {round(put_rho, 3)}
            </div>
            """,
                unsafe_allow_html=True,
            )

    with tab3:
        st.header("Heatmap")
        st.write(
            """
            Please adjust the follow inputs to analyse how the most sensitive inputs of
            volatility and underlying price affect option prices, whilst keeping all other factors constant.
        """
        )
        spot_min = st.number_input(
            "Min Spot Price", min_value=0.01, value=S * 0.9, step=0.01
        )
        spot_max = st.number_input(
            "Max Spot Price", min_value=0.01, value=S * 1.1, step=0.01
        )
        vol_min = st.number_input(
            "Min Volatility",
            min_value=0.01,
            max_value=1.0,
            value=sigma * 0.5,
            step=0.01,
        )
        vol_max = st.number_input(
            "Max Volatility",
            min_value=0.01,
            max_value=1.0,
            value=sigma * 1.5,
            step=0.01,
        )

        st.write("---")

        # Make the heatmap function
        def create_custom_cmap():
            """Create a custom colormap that transitions from red to green."""
            colors = ["red", "green"]
            cmap_name = "red_to_green"
            return LinearSegmentedColormap.from_list(cmap_name, colors)

        def plot_heatmap(spot_min, spot_max, vol_min, vol_max, r, X, t):
            # Define the resolution
            num_points = 100
            spot_prices = np.linspace(spot_min, spot_max, num_points)
            volatilities = np.linspace(vol_min, vol_max, num_points)
            call_prices = np.zeros((num_points, num_points))
            put_prices = np.zeros((num_points, num_points))

            for i, sigma in enumerate(volatilities):
                for j, S in enumerate(spot_prices):
                    bs_model = BlackScholes(r, X, S, t, sigma)
                    call_price, put_price = bs_model.compute()
                    call_prices[i, j] = call_price
                    put_prices[i, j] = put_price

            # Create custom colormap for green and red
            custom_cmap = create_custom_cmap()

            # Create the heatmaps
            fig, ax = plt.subplots(1, 2, figsize=(18, 8))

            # Heatmap number 1
            sns.heatmap(
                call_prices,
                ax=ax[0],
                cmap=custom_cmap,
                cbar_kws={"label": "Call Price"},
            )
            ax[0].set_title("Call Option Price Heatmap")
            ax[0].set_xlabel("Spot Price")
            ax[0].set_ylabel("Volatility")
            ax[0].set_xticks(np.linspace(0, num_points - 1, 10, dtype=int))
            ax[0].set_xticklabels(
                [
                    f"{spot_min + (spot_max - spot_min) * i / (num_points - 1):.2f}"
                    for i in np.linspace(0, num_points - 1, 10, dtype=int)
                ],
                rotation=0,
            )
            ax[0].set_yticks(np.linspace(0, num_points - 1, 10, dtype=int))
            ax[0].set_yticklabels(
                [
                    f"{vol_min + (vol_max - vol_min) * i / (num_points - 1):.2f}"
                    for i in np.linspace(0, num_points - 1, 10, dtype=int)
                ],
                rotation=0,
            )
            ax[0].invert_yaxis()  # Invert y-axis

            # Heatmap number 2
            sns.heatmap(
                put_prices, ax=ax[1], cmap=custom_cmap, cbar_kws={"label": "Put Price"}
            )
            ax[1].set_title("Put Option Price Heatmap")
            ax[1].set_xlabel("Spot Price")
            ax[1].set_ylabel("Volatility")
            ax[1].set_xticks(np.linspace(0, num_points - 1, 10, dtype=int))
            ax[1].set_xticklabels(
                [
                    f"{spot_min + (spot_max - spot_min) * i / (num_points - 1):.2f}"
                    for i in np.linspace(0, num_points - 1, 10, dtype=int)
                ],
                rotation=0,
            )
            ax[1].set_yticks(np.linspace(0, num_points - 1, 10, dtype=int))
            ax[1].set_yticklabels(
                [
                    f"{vol_min + (vol_max - vol_min) * i / (num_points - 1):.2f}"
                    for i in np.linspace(0, num_points - 1, 10, dtype=int)
                ],
                rotation=0,
            )
            ax[1].invert_yaxis()  # Invert y-axis

            plt.tight_layout()
            return fig

        # Generate and display heatmaps
        fig = plot_heatmap(spot_min, spot_max, vol_min, vol_max, r, X, t)
        st.pyplot(fig)

        st.write("---")
        st.subheader("P&L")
        st.write(
            "Now we can explore the P&L of a call option of your choice for different volatilities and underlying price, fixing the rest of the inputs."
        )

        # Add user inputs for P&L heatmap
        X_pnl = st.number_input(
            "Strike Price for P&L", min_value=0.01, value=100.0, step=0.01
        )
        t_pnl = st.number_input(
            "Time to Expiration (Years) for P&L", min_value=0.01, value=1.0, step=0.01
        )
        r_pnl = st.number_input(
            "Interest Rate for P&L", min_value=0.01, value=0.05, step=0.01
        )
        premium = st.number_input(
            "Option Premium (Cost)", min_value=0.01, value=10.0, step=0.01
        )  # User input for the option premium

        st.write("---")

        def create_custom_cmap1():
            """
            Create a custom colormap that transitions from red to green.
            Red is for negative values and green is for positive values.
            """
            colors = [
                "red",
                "white",
                "green",
            ]  # Red for negative, white for zero, green for positive
            cmap_name = "red_white_to_green"
            return LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

        def plot_pnl_heatmap_with_bins(
            spot_min, spot_max, vol_min, vol_max, r, X, t, premium, bins=10
        ):
            spot_bins = np.linspace(spot_min, spot_max, bins + 1)
            vol_bins = np.linspace(vol_min, vol_max, bins + 1)
            pnl_values = np.zeros((bins, bins))

            for i in range(bins):
                for j in range(bins):
                    spot_bin_center = (spot_bins[j] + spot_bins[j + 1]) / 2
                    vol_bin_center = (vol_bins[i] + vol_bins[i + 1]) / 2
                    bs_model = BlackScholes(r, X, spot_bin_center, t, vol_bin_center)
                    call_price, _ = bs_model.compute()
                    pnl_values[i, j] = call_price - premium  # Calculate P&L

            # Create the custom colormap
            custom_cmap1 = create_custom_cmap1()

            # Create the P&L heatmap with bins
            fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figsize here
            sns.heatmap(
                pnl_values,
                cmap=custom_cmap1,
                cbar_kws={"label": "P&L"},
                ax=ax,
                center=0,  # Center the colormap around zero
                xticklabels=[f"{spot_bins[j]:.2f}" for j in range(bins)],
                yticklabels=[f"{vol_bins[i]:.2f}" for i in range(bins)],
                annot=True,
                fmt=".2f",
                annot_kws={"size": 8},
            )  # Adjust annot_kws size if needed
            ax.set_title("Call Option P&L Heatmap")
            ax.set_xlabel("Spot Price")
            ax.set_ylabel("Volatility")
            ax.invert_yaxis()  # Invert y-axis

            st.pyplot(fig)

        plot_pnl_heatmap_with_bins(
            spot_min, spot_max, vol_min, vol_max, r_pnl, X_pnl, t_pnl, premium, bins=10
        )


if __name__ == "__main__":
    main()
