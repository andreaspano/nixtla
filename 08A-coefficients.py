import numpy as np
import pandas as pd
from scipy.stats import t

# Import StatsForecast and ARIMA models
from statsforecast import StatsForecast
from statsforecast.models import ARIMA, AutoARIMA
from statsforecast.utils import AirPassengersDF

def check(sf, data):
    """
    Fit a StatsForecast model to data and print ARIMA structure and coefficient statistics.

    Parameters
    ----------
    sf : StatsForecast
        A StatsForecast object with one or more models.
    data : pandas.DataFrame
        Time series data to fit the model(s) on.

    Prints
    ------
    - ARIMA model structure (order and seasonal order)
    - DataFrame with coefficient names, values, standard errors, degrees of freedom,
      t-statistics, and two-sided p-values for the null hypothesis that each coefficient is zero.
    """
    # Fit the model(s) to the data
    sf.fit(data)

    # Extract the fitted model parameters from the first model
    fitted_obj = sf.fitted_[0][0].model_
    
    # Get ARIMA order and seasonal order tuple
    arma = fitted_obj['arma']

    # Extract coefficient names and values
    coef = list(fitted_obj['coef'].keys())
    value = np.array(list(fitted_obj['coef'].values()))

    # Calculate standard errors from the diagonal of the variance-covariance matrix
    se = np.sqrt(np.diag(fitted_obj['var_coef']))

    # Degrees of freedom: number of observations minus number of estimated parameters
    df = fitted_obj.get('nobs', None) - len(coef)

    # t-statistics for each coefficient
    T = value / se
    # Two-sided p-values for each coefficient
    p_value = 2 * t.sf(np.abs(T), df)

    # Build a DataFrame summarizing coefficient statistics
    out = pd.DataFrame({
        'coef': coef,
        'value': value,
        'se': se,
        'df': df,
        'T': T,
        'p_value': p_value
    })

    # Build a DataFrame for the ARIMA order
    df_arma = pd.DataFrame([arma], columns=['p', 'q', 'P', 'Q', 'm', 'd', 'D'])
    
    # Print ARIMA order and coefficient summary
    
    print(f"Model Structure: \n {df_arma} \n") 
    print(f"Model Parameters: \n {out} \n") 

    return None


# Load example data
data = AirPassengersDF

# Define and fit AutoARIMA model
M0 = AutoARIMA(season_length=12)
sf0 = StatsForecast(
    models=[M0],
    freq='ME'
)

# Define and fit a specific ARIMA model
M1 = ARIMA(
    order=(1, 1, 0),
    season_length=12,
    seasonal_order=(0, 1, 0)
)
sf1 = StatsForecast(
    models=[M1],
    freq='ME'
)

# Run the check function for both models
check(sf0, data) # AutoARIMA
check(sf1, data) # ARIMA(1,1,0)(0,1,0)[12]


'''
(p, d, q, P, m, Q, D)  — SARIMA parameter meaning
---------------------------------------------------------------

Symbol | Name                     | What it controls
-------+--------------------------+----------------------------------------------
p      | Non-seasonal AR order    | # of autoregressive lags (y[t-1], y[t-2], ...)
d      | Non-seasonal differencing| # of regular differences to remove trend
q      | Non-seasonal MA order    | # of lagged forecast errors used
P      | Seasonal AR order        | # of AR terms at seasonal lags (m, 2m, ...)
m      | Seasonal period          | Length of season (observations per cycle)
Q      | Seasonal MA order        | # of MA terms at seasonal lags (m, 2m, ...)
D      | Seasonal differencing    | # of seasonal differences
'''
