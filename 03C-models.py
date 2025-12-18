from defs import *


h = 3
n_windows = 10
step_size = 3
min_lenght = 12
freq = 'MS'

# Read data
dtype={'year': int, 'month': object, 'unique_id': object, 'y':float, 'desc': float}
df = pd.read_csv('~/dev/ml/data/fondi-economics.csv', dtype=dtype)



# Prepaer data
df = prepare(df, min_length= min_lenght)

df = df[df["ds"] <= "2024-12-01"]


# Build the hierarchy 
df['Total'] = 'TOTAL'
spec = [
    ['Total'],               # top level (TOTAL)
    ['Total', 'series_id']   # bottoms (each series)
]
# Aggregate Series
df, S_df, tags = hf_aggregate(df=df, spec=spec)


# Split trn & tst
df_trn , df_tst = split(df, h = h)


# Model list
models = [
    AutoARIMA(),
    SeasonalNaive12(),
    HistoricAverage(),
    #HoltWinters12(),
    AutoETS(),
    #CrostonOptimized(),
    Naive()
    #AutoTBATS12()
]



## Instantiate StatsForecast class as sf
sf = StatsForecast( 
    models=models,
    freq='MS', 
    #fallback_model = SeasonalNaive(season_length=12),
    fallback_model = HistoricAverage(),
    n_jobs=-1,
)


# Find Best Model
best_model_df = best_model(df = df_trn, sf = sf, h = h, n_windows = n_windows, step_size = step_size)

# Forecast Best Model 
fitted_df, forecast_df = forecast_best(df_trn = df_trn, best_model_df = best_model_df, h = h, freq = freq)


# Reconcile 
forecast_df = reconcile_best(forecast_df = forecast_df, fitted_df = fitted_df, S_df = S_df, tags = tags)

# Mape Best Model
mape_best(forecast_df = forecast_df, df_tst = df_tst)

# Plot 
from plotnine import ggplot, aes, geom_line, facet_wrap 


merged_df = pd.merge(
        df_tst,          # test data with actual y
        forecast_df,     # forecasted data with y_hat
        on=["unique_id", "ds"],  # join keys
        how="left"       # keep all test rows (or "inner" if you only want matched ones)
    )


df_plot = df[df["unique_id"] != "TOTAL"]
df_plot = df_plot.groupby("unique_id").tail(12)
df_plot["unique_id"] = df_plot["unique_id"].str[-8:]
merged_df = merged_df[merged_df["unique_id"] != "TOTAL"]
merged_df["unique_id"] = merged_df["unique_id"].str[-8:]


(
    ggplot(df_plot)  # What data to use
        + aes(x="ds", y="y")  # What variable to use
        + geom_line() 
        + geom_line(aes(x="ds", y="yhat"), data=merged_df, color="red")
        + facet_wrap("unique_id", scales="free_y") # Geometric object to use for drawing
)



df_plot = df[df["unique_id"] != "TOTAL"]
df_plot = df_plot.groupby("unique_id").tail(36)
df_plot["unique_id"] = df_plot["unique_id"].str[-8:]
test = df_plot[df_plot["unique_id"] == "00001057"]

(
    ggplot(test)  # What data to use
        + aes(x="ds", y="y")  # What variable to use
        + geom_line() 
        #+ geom_line(aes(x="ds", y="yhat"), data=merged_df, color="red")
        + facet_wrap("unique_id", scales="free_y") # Geometric object to use for drawing
)

df_trn , df_tst = split(test, h = h)


models = [AutoARIMA()]

sf = StatsForecast(
    models = models, 
    freq = 'MS', 
    n_jobs = 1
)

cv_df = sf.cross_validation(
    df = df_trn,
    h = 3,
    step_size = 3,
    n_windows = 10
  )

cv_df.rename(columns = {'y' : 'actual'}, inplace = True) # rename actual values 

from utilsforecast.losses import rmse
from utilsforecast.losses import mape


cv_rmse = rmse(cv_df, models=['AutoARIMA'], target_col='actual')['AutoARIMA'].item()
print(f"RMSE using cross-validation: {cv_rmse:.2f}")

cv_mape = mape(cv_df, models=['AutoARIMA'], target_col='actual')['AutoARIMA'].item()
print(f"MAPE using cross-validation: {cv_mape:.2f}")

sf = sf.fit(df=df_trn)

forecasts = sf.predict(h=3)
forecasts.head()

merged_df = pd.merge(
        df_tst,          # test data with actual y
        forecasts,     # forecasted data with y_hat
        on=["unique_id", "ds"],  # join keys
        how="left"       # keep all test rows (or "inner" if you only want matched ones)
    )


from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, mape, mase, rmse, smape

eval_df = evaluate(
    df= merged_df,
    train_df=df_trn,
    metrics=[mae, mape, rmse, smape],
    agg_fn='mean',
).set_index('metric').T
eval_df