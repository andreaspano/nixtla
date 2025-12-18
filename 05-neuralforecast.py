import logging

import numpy as np
import pandas as pd
from utilsforecast.plotting import plot_series

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS
from neuralforecast.utils import AirPassengersDF

logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

#############################
# Imports


# functions
def parse_italian_month(s):
    s = s.replace('.', '')  # remove the dot
    for k, v in month_map.items():
        if k in s:
            year = s[-2:]
            return pd.to_datetime(f'01-{v}-20{year}', format='%d-%m-%Y', errors='coerce')
    return pd.NaT


def MAPE(y, y_hat):
    mask = y != 0
    return np.mean(np.abs((y[mask] - y_hat[mask]) / y[mask]))


def cross_validation_stats(cv_df):
    
    cv_df = (
        cv_df
        .melt(
            id_vars=['unique_id', 'cutoff', 'y', 'ds'],   # columns to keep fixed
            var_name='model',                       # name of the new “model” column
            value_name='y_hat'                      # name for model forecast values
            )
        )

    
    
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

    return mape_stats
    

#########################################
# Read data
dtype={'year': int, 'month': object, 'unique_id': object, 'y':float, 'desc': float}
tmp = pd.read_csv('~/dev/ml/data/fondi-economics.csv', dtype=dtype)

#########################################

# Data preparation
## Convert dates
month_map = {
    'Gen': '01',
    'Feb': '02',
    'Mar': '03',
    'Apr': '04',
    'Mag': '05',
    'Giu': '06',
    'Lug': '07',
    'Ago': '08',
    'Set': '09',
    'Ott': '10',
    'Nov': '11',
    'Dic': '12'
}



tmp['ds'] = tmp['Mese'].apply(parse_italian_month)

## aggregate
tmp['unique_id'] = tmp['Key VSPC'].str[:13]
df = tmp.groupby(['unique_id', 'ds'], as_index=False)['Valore'].sum()    

## rename
df = df.rename(columns={'Valore':'y'})

## Check obs number
series_length = (
    df
        .groupby(['unique_id'], as_index=False)
        .size()
        .sort_values('size', ascending=True) 
)

# flter long series > 12
long_series = series_length[series_length['size'] >= 12].unique_id
df = df[df['unique_id'].isin(long_series)]
#########################################
# Plot Discovery
#StatsForecast.plot(df).savefig('./figs/pl_03_01.png')

#############################

h = 12

# Split data and declare panel dataset
Y_df = df
Y_train_df = Y_df[Y_df.ds<'2024-05-01'] # 132 train
Y_test_df = Y_df[Y_df.ds>='2024-05-01'] # 12 test

# Fit and predict with NBEATS and NHITS models
horizon = 12
models = [NBEATS(input_size=2 * horizon, h=horizon, max_steps=100, enable_progress_bar=False),
          NHITS(input_size=2 * horizon, h=horizon, max_steps=100, enable_progress_bar=False)]
nf = NeuralForecast(models=models, freq='ME')
nf.fit(df=Y_train_df)
Y_hat_df = nf.predict()

# Plot predictions
plot_series(Y_df, Y_hat_df).savefig('./figs/pl_05_01.png')