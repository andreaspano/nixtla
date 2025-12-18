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

from plotnine import ggplot, aes, geom_line, facet_wrap 


# Prepare data
# need to be arranged for each dataset 
def prepare(df):
    """
    Prepare input dataframe for hierarchical forecasting.

    Actions:
      - Ensures 'ds' is datetime.
      - Adds a top-level 'Total' column with value 'TOTAL'.
      - Builds hierarchical aggregation spec and returns aggregated frames.

    Args:
      df (pd.DataFrame): input with at least columns ['unique_id','ds','y','id'].

    Returns:
      tuple: (df_aggregated (pd.DataFrame), S_df (pd.DataFrame), tags (dict))
    """
    # ensure datetime type for the time column
    df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d')

    # Build the hierarchy at runtime: a single TOTAL top level and bottoms by id
    df['Total'] = 'TOTAL'
    spec = [
        ['Total'],               # top level (TOTAL)
        ['Total', 'id']          # bottom level (each series)
    ]

    # Aggregate series according to spec -> returns aggregated df, S matrix, and tags
    df, S_df, tags = hf_aggregate(df=df, spec=spec)

    return df, S_df, tags


# init models
def init_model(model_dict, freq):
    """
    Initialize a StatsForecast instance from a mapping of model aliases to model instances.

    Side-effects:
      - Sets model.alias = alias for every model in model_dict so the model outputs
        are tagged with readable names.

    Args:
      model_dict (dict): mapping alias (str) -> model instance (e.g. AutoARIMA(...)).
      freq (str): frequency string expected by StatsForecast (e.g. 'MS', 'D').

    Returns:
      StatsForecast: instantiated StatsForecast configured with the provided models.
    """
    # Ensure every model carries its alias for clear identification in CV / forecasts.
    for alias, model in model_dict.items():
        model.alias = alias

    # Instantiate StatsForecast with the provided models.
    # - fallback_model: used when a model fails for a series
    # - n_jobs: parallelism (-1 uses all available cores)
    sf = StatsForecast(
        models=list(model_dict.values()),
        freq=freq,
        fallback_model=HistoricAverage(),
        n_jobs=-1,
    )

    return sf



# best model
def best_model(df, sf, h, n_windows, step_size):
    """
    Run cross-validation with a StatsForecast instance and select the best model per series.

    Steps:
      - Runs sf.cross_validation on df with provided horizon and windows.
      - Computes cross-validation statistics (MAPE-based).
      - Picks per-unique_id the model with minimum mape_L2.

    Args:
      df (pd.DataFrame): training dataframe used for cross-validation.
      sf (StatsForecast): configured StatsForecast instance (models must have .alias).
      h (int): forecast horizon used in CV.
      n_windows (int): number of CV windows.
      step_size (int): CV step size (months/periods).

    Returns:
      tuple: (df_best_model (pd.DataFrame), df_cv_stat (pd.DataFrame))
        - df_best_model: one row per unique_id with chosen model.
        - df_cv_stat: per-model CV statistics used for selection.
    """
    ## Cross Validation
    df_cv = sf.cross_validation(
        df=df,
        h=h,
        step_size=step_size,
        n_windows=n_windows
    )

    # Cross validation Stats
    df_cv_stat = cross_validation_stats(df_cv)

    # Best Model per series by mape_L2
    df_best_model = (
        df_cv_stat
            .loc[df_cv_stat.groupby('unique_id')['mape_L2'].idxmin()]
            .reset_index(drop=True)
    )

    return df_best_model, df_cv_stat




# Split trn & tst
# Split
def split(df, h):
    """
    Split dataframe into training and test sets by time.

    Args:
      df (pd.DataFrame): dataframe with 'ds' datetime column.
      h (int): number of periods (months) to keep as test.

    Returns:
      tuple: (df_trn (pd.DataFrame), df_tst (pd.DataFrame))
    """
    # compute cutoff timestamp by subtracting h months from the max date
    ds_split = df['ds'].max() -  pd.DateOffset(months=h)
    df_trn =  df[df['ds'] <= ds_split]
    df_tst =  df[df['ds'] > ds_split]

    return df_trn, df_tst


# MAPE
def MAPE(y, y_hat):
    """
    Mean Absolute Percentage Error avoiding division by zero.

    Args:
      y (array-like): actual values.
      y_hat (array-like): predicted values.

    Returns:
      float: mean absolute percentage error computed only for y != 0.
    """
    mask = y != 0
    return np.mean(np.abs((y[mask] - y_hat[mask]) / y[mask]))


# Cross validation kpi
def cross_validation_stats(cv_df):
    """
    Compute per-series, per-model MAPE statistics from cross-validation output.

    Steps:
      - Melt CV dataframe to (unique_id, cutoff, y, ds, model, y_hat).
      - Compute MAPE per (unique_id, model, cutoff).
      - Aggregate to mape_mean, mape_std, mape_max and compute mape_L2.

    Args:
      cv_df (pd.DataFrame): cross-validation dataframe returned by StatsForecast.cross_validation.

    Returns:
      pd.DataFrame: aggregated MAPE statistics per unique_id and model with computed mape_L2.
    """
    cv_df = (
        cv_df
        .melt(
            id_vars=['unique_id', 'cutoff', 'y', 'ds'],   # columns to keep fixed
            var_name='model',                             # name of the new “model” column
            value_name='y_hat'                            # name for model forecast values
            )
        )

    # compute MAPE per fold (unique_id, model, cutoff)
    mape_df = (
        cv_df.groupby(['unique_id', 'model', 'cutoff'], as_index=False)
        .apply(lambda g: pd.Series({'mape': MAPE(g['y'], g['y_hat'])}), include_groups=False)
        .reset_index(drop=True)
    )

    # aggregate across cutoffs for each model-series pair
    mape_stats = (
        mape_df
        .groupby(['unique_id', 'model'], as_index=False)
        .agg(
            mape_mean=('mape', 'mean'),
            mape_std=('mape', 'std'),
            mape_max=('mape', 'max')
        )
    )

    # combine mean and variability into a single L2-style metric used for selection
    mape_stats = mape_stats.assign(
        mape_L2 = np.sqrt(
            mape_stats['mape_mean']**2 +
            mape_stats['mape_std']**2
            # optionally include max-based term if needed
        )
    )

    return mape_stats
    

# make model
def make_model(name: str, model_dict):
    """
    Return a model instance from a model dictionary by name.

    Args:
      name (str): alias/key for model in model_dict.
      model_dict (dict): mapping alias -> model instance.

    Returns:
      model instance: the model object (not fitted).
    """
    best_model = model_dict[name]
    return best_model
    



# Forecast Best Model
def forecast_best(df_trn , df_best_model, model_dict, h , freq):
    """
    Fit and forecast each series with its selected best model.

    Steps per series:
      - Instantiate the selected model.
      - Fit using StatsForecast on the single-series dataframe.
      - Retrieve h-step forecasts and in-sample fitted values.
      - Normalize forecast/fitted column names to 'yhat'.

    Args:
      df_trn (pd.DataFrame): training dataframe with columns ['unique_id','ds','y'].
      df_best_model (pd.DataFrame): dataframe with columns ['unique_id','model'] selecting model per series.
      model_dict (dict): mapping model alias -> model instance.
      h (int): forecast horizon.
      freq (str): frequency string for StatsForecast.

    Returns:
      tuple: (df_fit (pd.DataFrame), df_fct (pd.DataFrame))
        - df_fit: in-sample actuals + fitted values with columns ['unique_id','ds','y','yhat'].
        - df_fct: future forecasts with columns ['unique_id','ds','yhat'].
    """
    forecasts = []
    fitted_values = []

    for _, row in df_best_model.iterrows():
        uid = row["unique_id"]
        model_name = row["model"]

        # instantiate the correct model (object from model_dict)
        m = make_model(model_name, model_dict)

        # subset this series for fitting/forecasting
        ser = df_trn.loc[df_trn["unique_id"] == uid, ["unique_id", "ds", "y"]]

        # fit & forecast on single-series with StatsForecast
        sf = StatsForecast(models=[m], freq=freq, n_jobs=1)
        fcst = sf.forecast(df=ser, h=h, fitted=True).reset_index(drop=True)
        fit = sf.forecast_fitted_values().reset_index(drop=True)

        # find the forecast column name created by the model and normalize to 'yhat'
        pred_col = [c for c in fcst.columns if c not in {"unique_id", "ds"}][0]
        fcst = fcst.rename(columns={pred_col: "yhat"})
        fcst["model"] = model_name

        forecasts.append(
            fcst[["unique_id", "ds", "yhat"]]  # only keep standardized columns
        )

        # normalize fitted column to 'yhat' as well
        fit_col = [c for c in fit.columns if c not in {"unique_id", "ds", "y"}][0]
        fit = fit.rename(columns={fit_col: "yhat"})

        fitted_values.append(
            fit[["unique_id", "ds", "y", "yhat"]]
        )

    # combine results after loop
    df_fct = pd.concat(forecasts, ignore_index=True)      # future forecasts
    df_fit = pd.concat(fitted_values, ignore_index=True)  # in-sample actuals + fitted

    return df_fit, df_fct

    


# Reconcile 
def reconcile_best(df_fct, df_fit, S_df, tags):
    """
    Reconcile bottom-level forecasts to the hierarchy using hierarchicalforecast.

    Args:
      df_fct (pd.DataFrame): base forecasts with columns ['unique_id','ds','yhat'].
      df_fit (pd.DataFrame): in-sample actuals + fitted values used to compute S matrix if needed.
      S_df (pd.DataFrame): summing matrix returned by hf_aggregate.
      tags (dict): tags describing aggregation mapping returned by hf_aggregate.

    Returns:
      pd.DataFrame: reconciled forecasts; column 'yhat_rec' contains reconciled values.
    """
    # choose reconciler (mint_shrink chosen as more stable than OLS)
    hrec = HierarchicalReconciliation(reconcilers=[MinTrace(method='mint_shrink')])

    df_fct = hrec.reconcile(
        Y_hat_df=df_fct,  # base forecasts (bottoms; optionally top)
        Y_df=df_fit,
        S_df=S_df,
        tags=tags
    )

    # rename reconciled column (the reconciler returns an extra forecast column)
    col = [c for c in df_fct.columns if c not in {'unique_id', 'ds', 'yhat'}][0]
    df_fct = df_fct.rename(columns={col:'yhat_rec'})

    return df_fct

# Mape on best model
def mape_best (df_fct, df_tst):
    """
    Compute MAPE and reconciled MAPE for test set forecasts.

    Args:
      df_fct (pd.DataFrame): forecasts containing 'yhat' and 'yhat_rec' for each unique_id/ds.
      df_tst (pd.DataFrame): test dataframe with actual 'y' values.

    Returns:
      pd.DataFrame: per-unique_id dataframe with columns ['unique_id','mape','mape_rec'].
    """
    merged_df = pd.merge(
        df_tst,          # test data with actual y
        df_fct,          # forecasted data with y_hat and yhat_rec
        on=["unique_id", "ds"],  # join keys
        how="left"       # keep all test rows
    )

    # compute MAPE and reconciled MAPE per series
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

# Plot results
def plot_result(df, df_fct , df_tst, n_tail, total = False):
    """
    Create a faceted time-series plot with actuals and forecasts.

    Behavior:
      - Merges test and forecast frames to get forecasted values per timestamp.
      - Optionally excludes 'TOTAL' series from plotting.
      - Keeps the last n_tail observations per series for visualization.
      - Returns a plotnine (ggplot) object.

    Args:
      df (pd.DataFrame): original (possibly aggregated) dataframe with 'y'.
      df_fct (pd.DataFrame): forecast dataframe with 'yhat' (and possibly 'yhat_rec').
      df_tst (pd.DataFrame): test dataframe used to align forecasts by ds.
      n_tail (int): number of latest observations per series to show.
      total (bool): if True include TOTAL in plotting; otherwise exclude it.

    Returns:
      plotnine.ggplot: faceted plot object.
    """
    # Merge test and forecasts on unique_id & ds to obtain forecast traces
    df_merged = pd.merge(
            df_tst,
            df_fct,
            on=["unique_id", "ds"],
            how="left"
        )

    # Exclude TOTAL series unless requested
    if total == True:
        df_plot = df
    else:
        df_plot = df[df["unique_id"] != "TOTAL"]
        df_merged = df_merged[df_merged["unique_id"] != "TOTAL"]

    # keep the last n_tail points per series for plot clarity
    df_plot = df_plot.groupby("unique_id").tail(n_tail)

    # build plotnine object: actuals in default color, forecasts in red
    plt = (
        ggplot(df_plot)  # dataset for actuals
            + aes(x="ds", y="y")
            + geom_line()
            + geom_line(aes(x="ds", y="yhat"), data=df_merged, color="red")
            + facet_wrap("unique_id", scales="free_y")
    )

    return(plt)
