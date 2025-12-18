
# Simulated Data
from defs02E import * 


# Params
h = 12 #Forecast horizon 
n_windows = h # number of cv windows
step_size = 1 # step size between cv windows
freq = 'MS' # frequency of data

# Models
model_dict = {
    "AutoARIMA12": AutoARIMA(season_length=12),
    "AutoARIMA01": AutoARIMA(season_length=1),
    "HistoricAverage": HistoricAverage(),
    "AutoETS12": AutoETS(season_length=12),
    "AutoETS01": AutoETS(season_length=1),
}


#read data
df = pd.read_csv('~/dev/ml/data/sim.csv')


# data preparation
# require ad hoc def
df, S_df, tags = prepare(df)

# Split trn & tst
df_trn, df_tst = split(df, h)



# Init Model
sf =  init_model(model_dict = model_dict, freq = freq)
    
# Find Best Model
df_best_model, df_cv_stat = best_model(df = df_trn, sf = sf,  h = h, n_windows = n_windows, step_size = step_size)


# Forecast Best Model 
df_fit, df_fct = forecast_best(df_trn = df_trn, df_best_model = df_best_model, model_dict = model_dict, h = h, freq = freq)

# Reconcile 
df_rec = reconcile_best(df_fct = df_fct, df_fit = df_fit, S_df = S_df, tags = tags)

# Mape Best Model
mape_best(df_fct = df_rec, df_tst = df_tst)

# Plot 
plot_result(df = df, df_fct = df_fct , df_tst = df_tst, n_tail = 36, total = True)



