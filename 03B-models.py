# Imports
import pandas as pd
import numpy as np
import inspect

from statsforecast import StatsForecast
import statsforecast.models as sfm
from statsforecast.models import (
    HistoricAverage,
    SeasonalNaive,
    AutoARIMA,
    HoltWinters,
    AutoETS,
    CrostonOptimized,
    GARCH,
    Naive,
    AutoTBATS
)
from hierarchicalforecast.methods import BottomUp, MinTrace
from hierarchicalforecast.utils import aggregate, HierarchicalPlot
from hierarchicalforecast.core import HierarchicalReconciliation

# hyper paramemetets
h = 3
s = 12
n_windows=10
min_length = 12


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
            mape_std=('mape', 'std'),
            mape_max=('mape', 'max')
        )
    )

    mape_stats = mape_stats.assign(
        mape_L2 = np.sqrt(
            0.5*mape_stats['mape_mean']**2 + 
            0.4*mape_stats['mape_std']**2 + 
            0.1*mape_stats['mape_max']**2))

    return mape_stats
    
# def seasonalNaive defaulst to 12

def SeasonalNaive12(alias="SeasonalNaive12"):
    return SeasonalNaive(season_length=12, alias=alias)

sfm.SeasonalNaive12 = SeasonalNaive12  # inject into module    


  
# def HoltWinters default to 12

def HoltWinters12(alias="HoltWinters12"):
    return HoltWinters(season_length=12, alias=alias)

sfm.HoltWinters12 = HoltWinters12  # inject into module    

 
# def AutoTBATS default to 12

def AutoTBATS12(alias="AutoTBATS12"):
    return AutoTBATS(season_length=12, alias=alias)

sfm.AutoTBATS12 = AutoTBATS12  # inject into module    



def make_model(name: str, params=None):
    """Return an instantiated model given its name and optional params."""
    params = params or {}
    if not hasattr(sfm, name):
        raise ValueError(f"Unknown model '{name}' in statsforecast.models")
    obj = getattr(sfm, name)
    if inspect.isclass(obj) or callable(obj):
        return obj(**params)
    raise TypeError(f"'{name}' is not a class or callable")

#########################################
## Define models
models = [
    AutoARIMA(),
    SeasonalNaive12(),
    HistoricAverage(),
    HoltWinters12(),
    AutoETS(),
    #CrostonOptimized(),
    GARCH(),
    Naive(),
    AutoTBATS12()
]


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
df = df.rename(columns={'Valore':'y', 'unique_id':'series_id'})

## Check obs number
series_length = (
    df
        .groupby(['series_id'], as_index=False)
        .size()
        .sort_values('size', ascending=True) 
)

# flter long series > 12

long_series = series_length[series_length['size'] >= min_length].series_id
df = df[df['series_id'].isin(long_series)]
df = df.copy()
df['All'] = 'TOTAL'  

spec = [
    ['All'],                # top level (TOTAL)
    ['All', 'series_id']    # bottom level (each unique_id under TOTAL)
]
Y_df, S_df, tags = aggregate(df=df, spec=spec) 
df =  Y_df

# split train and test
ds_split = df['ds'].max() -  pd.DateOffset(months=h)
df_trn =  df[df['ds'] <= ds_split]
df_tst =  df[df['ds'] > ds_split]




#########################################
# Plot Discovery
StatsForecast.plot(df, plot_random=False).savefig('./figs/pl_03_01.png')

#########################################



## Instantiate StatsForecast class as sf
sf = StatsForecast( 
    models=models,
    freq='MS', 
    #fallback_model = SeasonalNaive(season_length=12),
    fallback_model = SeasonalNaive12(),
    n_jobs=-1,
)


## Cross Validation
cv_df = sf.cross_validation(
    df=df_trn,
    h=h,
    step_size=1,
    n_windows=n_windows
)

cv_stat_df = cross_validation_stats(cv_df)


###############################################



best_model_df = (
    cv_stat_df
        .loc[cv_stat_df.groupby('unique_id')['mape_L2'].idxmin()]
        .reset_index(drop=True)
)


################################################


forecasts = []

for _, row in best_model_df.iterrows():
    uid = row["unique_id"]
    model_name = row["model"]

    # instantiate correct model
    m = make_model(model_name)

    # subset this series
    ser = df_trn.loc[df_trn["unique_id"] == uid, ["unique_id", "ds", "y"]]

    # fit & forecast
    sf = StatsForecast(models=[m], freq="MS", n_jobs=1)
    fcst = sf.forecast(df=ser, h=h, fitted = True).reset_index(drop=True)
    fit = sf.forecast_fitted_values()

    # rename prediction column
    pred_col = [c for c in fcst.columns if c not in {"unique_id", "ds"}][0]
    fcst = fcst.rename(columns={pred_col: "y_hat"})
    fcst["model"] = model_name
    forecasts.append(fcst[["unique_id", "ds", "y_hat", "model"]])

forecast_df = pd.concat(forecasts, ignore_index=True)
print(forecast_df)

# rec

hrec.reconcile(Y_hat_df=fcst, Y_df=fit, 
                          S_df=S_df, tags=tags, level=[80, 90])


merged_df = pd.merge(
    df_tst,          # test data with actual y
    forecast_df,     # forecasted data with y_hat
    on=["unique_id", "ds"],  # join keys
    how="left"       # keep all test rows (or "inner" if you only want matched ones)
)


merged_mape_df = (
        merged_df.groupby(['unique_id', 'model'], as_index=False)
        .apply(lambda g: pd.Series({'mape': MAPE(g['y'], g['y_hat'])}), include_groups=False)
        .reset_index(drop=True)
    )


print(merged_mape_df)

##########################################
# Reconciliation
from hierarchicalforecast.methods import BottomUp, MinTrace
from hierarchicalforecast.utils import aggregate, HierarchicalPlot
from hierarchicalforecast.core import HierarchicalReconciliation
reconcilers = [
    BottomUp(),
    MinTrace(method='mint_shrink'),
    MinTrace(method='ols')
]

hrec = HierarchicalReconciliation(reconcilers=reconcilers)


Y_rec_df = hrec.reconcile(Y_hat_df=Y_hat_df, Y_df=Y_fitted_df, 
                          S_df=S_df, tags=tags, level=[80, 90])


hrec.reconcile(Y_hat_df=, Y_df=Y_fitted_df, 
                          S_df=S_df, tags=tags, level=[80, 90])