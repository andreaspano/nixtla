import pandas as pd
import numpy as np
import inspect

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
from hierarchicalforecast.utils import  HierarchicalPlot
from hierarchicalforecast.core import HierarchicalReconciliation

from statsforecast.core import StatsForecast
from hierarchicalforecast.utils import aggregate as hf_aggregate
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import MinTrace


# Prepare data
# need to be arranged for each dataset 

def prepare(df):
    df = df[["Code","Year","GDP"]]
    df = df.rename(columns={'Code':'series_id', 'Year': 'ds', 'GDP': 'y'})
    df = df.dropna(subset=['y'])
    df['ds'] = pd.to_datetime(df['ds'], format='%Y')


    # Build the hierarchy 
    df['Total'] = 'TOTAL'
    spec = [
        ['Total'],               # top level (TOTAL)
        ['Total', 'series_id']   # bottoms (each series)
    ]
    # Aggregate Series
    df, S_df, tags = hf_aggregate(df=df, spec=spec)

    #return 
    return df, S_df, tags
    




# best model
def best_model(df, sf, h, n_windows, step_size):

    ## Cross Validation
    cv_df = sf.cross_validation(
        df=df,
        h=h,
        step_size=1,
        n_windows=n_windows
    )

    # Cross validation Stats
    cv_stat_df = cross_validation_stats(cv_df)

    # Best Model
    best_model_df = (
        cv_stat_df
            .loc[cv_stat_df.groupby('unique_id')['mape_L2'].idxmin()]
            .reset_index(drop=True)
    )

    return best_model_df




# Split trn & tst
def split(df, h):
    ds_split = df['ds'].max() -  h
    df_trn =  df[df['ds'] < ds_split]
    df_tst =  df[df['ds'] >= ds_split]

    return df_trn, df_tst


# MAPE
def MAPE(y, y_hat):
    mask = y != 0
    return np.mean(np.abs((y[mask] - y_hat[mask]) / y[mask]))

# Cross validation kpi
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
            mape_stats['mape_mean']**2 + 
            mape_stats['mape_std']**2  
            #0.1*mape_stats['mape_max']**2
            )
    )

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


# def AutoTBATS default to 1
def AutoTBATS1(alias="AutoTBATS1"):
    return AutoTBATS(season_length=1, alias=alias)

sfm.AutoTBATS1 = AutoTBATS1  # inject into module    

# make model
# review it
'''
def make_model(name: str, params=None):
    """Return an instantiated model given its name and optional params."""
    params = params or {}
    if not hasattr(sfm, name):
        raise ValueError(f"Unknown model '{name}' in statsforecast.models")
    obj = getattr(sfm, name)
    if inspect.isclass(obj) or callable(obj):
        return obj(**params)
    raise TypeError(f"'{name}' is not a class or callable")
'''

def make_model(name: str, model_dict):
    best_model = model_dict[name]
    return best_model
    



# Forecast Best Model
def forecast_best(df_trn , best_model_df, model_dict, h , freq):

    forecasts = []
    fitted_values = []

    for _, row in best_model_df.iterrows():
        uid = row["unique_id"]
        model_name = row["model"]

        # instantiate the correct model
        m = make_model(model_name, model_dict)

        # subset this series 
        ser = df_trn.loc[df_trn["unique_id"] == uid, ["unique_id", "ds", "y"]]
        print(uid)
        print(m)
        

        # fit & forecast
        sf = StatsForecast(models=[m], freq=freq, n_jobs=1)
        fcst = sf.forecast(df=ser, h=h, fitted=True).reset_index(drop=True)
        fit = sf.forecast_fitted_values().reset_index(drop=True)

        # find the forecast column name (the model output)
        pred_col = [c for c in fcst.columns if c not in {"unique_id", "ds"}][0]

        fcst = fcst.rename(columns={pred_col: "yhat"})
        fcst["model"] = model_name

        forecasts.append(
            fcst[["unique_id", "ds", "yhat"]]  # we don't need model downstream
        )

        # Fitted values (in-sample) 
        fit_col = [c for c in fit.columns if c not in {"unique_id", "ds", "y"}][0]
        fit = fit.rename(columns={fit_col: "yhat"})
     

        fitted_values.append(
            fit[["unique_id", "ds", "y", "yhat"]]
        )

    # combine results after loop
    Y_hat_df = pd.concat(forecasts, ignore_index=True)      # future forecasts
    Y_df     = pd.concat(fitted_values, ignore_index=True)  # in-sample actuals + fitted

    return  Y_df , Y_hat_df


# Reconcile 
def reconcile_best(forecast_df, fitted_df, S_df, tags):

    #hrec = HierarchicalReconciliation(reconcilers=[MinTrace(method='ols')])
    hrec = HierarchicalReconciliation(reconcilers=[MinTrace(method='mint_shrink')])


    forecast_df = hrec.reconcile(
        Y_hat_df=forecast_df,  # base forecasts (bottoms; optionally top)
        Y_df=fitted_df,
        S_df=S_df,
        tags=tags
    )

    # rename  recocciled column to standard 'yhat'
    col = [c for c in forecast_df.columns if c not in {'unique_id', 'ds', 'yhat'}][0]
    forecast_df = forecast_df.rename(columns={col:'yhat_rec'})

    return forecast_df

# Mape on best model
def mape_best (forecast_df, df_tst):

    merged_df = pd.merge(
        df_tst,          # test data with actual y
        forecast_df,     # forecasted data with y_hat
        on=["unique_id", "ds"],  # join keys
        how="left"       # keep all test rows (or "inner" if you only want matched ones)
    )


    mape_df = (
        merged_df.groupby('unique_id', as_index=False)
        .apply(
            lambda g: pd.Series({
                'mape': MAPE(g['y'], g['yhat']),
                'mape_rec': MAPE(g['y'], g['yhat_rec'])
            }),
            include_groups=False
        )
        .reset_index(drop=True)
    )

    return mape_df


