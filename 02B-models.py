# Ref https://nixtlaverse.nixtla.io/statsforecast/docs/getting-started/getting_started_complete.html

from defs import * 
from plotnine import ggplot, aes, geom_line, facet_wrap 


h = 2
n_windows = 10
step_size = 1
freq = 1


#read data
df = pd.read_csv('~/dev/ml/data/global.csv')

# data preparation
df = df[["Code","Year","GDP"]]
df = df.rename(columns={'Code':'series_id', 'Year': 'ds', 'GDP': 'y'})
df = df.dropna(subset=['y'])
df.copy()['ds'] = pd.to_datetime(df['ds'], format='%Y')


# Build the hierarchy 
df['Total'] = 'TOTAL'
spec = [
    ['Total'],               # top level (TOTAL)
    ['Total', 'series_id']   # bottoms (each series)
]
# Aggregate Series
df, S_df, tags = hf_aggregate(df=df, spec=spec)


# Split trn & tst
ds_split = df['ds'].max() -  h
df_trn =  df[df['ds'] <= ds_split]
df_tst =  df[df['ds'] > ds_split]

# Models
models = [
    AutoARIMA(),
    HistoricAverage(),
    AutoETS(),
    AutoTBATS1(),
    Naive()
]




## Instantiate StatsForecast class as sf
sf = StatsForecast( 
    models=models,
    freq=freq, 
    fallback_model = HistoricAverage(),
    n_jobs=-1,
)


# Find Best Model
df_best_model = best_model(df = df_trn, sf = sf, h = h, n_windows = n_windows, step_size = step_size)



# Forecast Best Model 
df_fit, df_fct = forecast_best(df_trn = df_trn, best_model_df = df_best_model, h = h, freq = freq)


# Reconcile 
df_rec = reconcile_best(forecast_df = df_fct, fitted_df = df_fit, S_df = S_df, tags = tags)


# Mape Best Model
mape_best(forecast_df = df_rec, df_tst = df_tst)

# Plot 


df_merged = pd.merge(
        df_tst,          # test data with actual y
        df_fct,     # forecasted data with y_hat
        on=["unique_id", "ds"],  # join keys
        how="left"       # keep all test rows (or "inner" if you only want matched ones)
    )


df_plot = df[df["unique_id"] != "TOTAL"]
df_plot = df_plot.groupby("unique_id").tail(10)
df_merged = df_merged[df_merged["unique_id"] != "TOTAL"]


(
    ggplot(df_plot)  # What data to use
        + aes(x="ds", y="y")  # What variable to use
        + geom_line() 
        + geom_line(aes(x="ds", y="yhat"), data=df_merged, color="red")
        + facet_wrap("unique_id", scales="free_y") # Geometric object to use for drawing
)


