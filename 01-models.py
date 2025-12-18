# Ref https://nixtlaverse.nixtla.io/statsforecast/docs/getting-started/getting_started_complete.html

import pandas as pd
import numpy as np

from statsforecast import StatsForecast

from statsforecast.models import (
    HoltWinters,
    CrostonClassic as Croston, 
    HistoricAverage,
    DynamicOptimizedTheta as DOT,
    SeasonalNaive
)


Y_df = pd.read_parquet('https://datasets-nixtla.s3.amazonaws.com/m4-hourly.parquet')

# Select 10 ids to make the example faster
uids = Y_df['unique_id'].unique()[:2] 
Y_df = Y_df[Y_df['unique_id'].isin(uids)]

#Select last 7 days of data to make example faster
Y_df = Y_df.groupby('unique_id').tail(7 * 24) 

# Plot data
StatsForecast.plot(Y_df).savefig('./figs/pl01.png')

# Define models
models = [
    HoltWinters(),
    Croston(),
    SeasonalNaive(season_length=24),
    HistoricAverage(),
    DOT(season_length=24)
]

# Instantiate StatsForecast class as sf
sf = StatsForecast( 
    models=models,
    freq=1, 
    fallback_model = SeasonalNaive(season_length=7),
    n_jobs=-1,
)

# Do forecast
forecasts_df = sf.forecast(df=Y_df, h=3, level=[90])
forecasts_df.head()


# plot results
#sf.plot(Y_df,forecasts_df).savefig('./figs/pl02.png')

# Plot to unique_ids and some selected models
#sf.plot(Y_df, 
#    forecasts_df, 
#    models=["HoltWinters","DynamicOptimizedTheta"], 
#    unique_ids=["H10", "H105"], level=[90]).savefig('./figs/pl03.png')

# Cross Validation
cv_df = sf.cross_validation(
    df=Y_df,
    h=3,
    step_size=1,
    n_windows=4
)


cv_df = (
    cv_df
    .melt(
        id_vars=['unique_id', 'cutoff', 'y', 'ds'],   # columns to keep fixed
        var_name='model',                       # name of the new “model” column
        value_name='y_hat'                      # name for model forecast values
    )
)

#cv_df = cv_df[ ["unique_id", "y","HoltWinters", "cutoff"]]
#cv_df = cv_df.rename(columns = {"HoltWinters": "y_hat"})

###############################################
def MAPE(y, y_hat):
    mask = y != 0
    return np.mean(np.abs((y[mask] - y_hat[mask]) / y[mask]))

mape_df = (
    cv_df.groupby(['unique_id', 'model', 'cutoff'], as_index=False)
      .apply(lambda g: pd.Series({'mape': MAPE(g['y'], g['y_hat'])}), include_groups=False)
      .reset_index(drop=True)
)




mape_stats = (
    mape_df
    .groupby(['unique_id', 'model'], as_index=False)
    .agg(
        mape_mean=('mape', 'mean'),
        mape_std=('mape', 'std')
    )
)
