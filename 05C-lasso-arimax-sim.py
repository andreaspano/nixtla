
# Simulated Data
from defs05C import * 


# Params
h = 12 #Forecast horizon 
n_windows = h # number of cv windows
step_size = 1 # step size between cv windows
freq = 'MS' # frequency of data

# Models
model_dict = {
    "AutoARIMA12": AutoARIMA(season_length=12)
}



#read data
df_all = pd.read_csv('~/dev/ml/data/sim-exo.csv')

df_all = df_all[df_all["id"] == "A2"]
df_all = df_all.rename(columns={"id": "unique_id"})

df_all['ds'] = pd.to_datetime(df_all['ds'], format='%Y-%m-%d')




# data preparation
# require ad hoc def
#exog_vars = {"x1": ["x1"], "x2": ["x2"], "x3": ["x3"]}

#df, S_df, tags = prepare(df_all)

# Split trn & tst
df_trn, df_tst = split(df_all, h)

##########################################


# Init Model
sf =  init_model(model_dict = model_dict, freq = freq)
    

# Fit Model
sf.fit(df = df_trn)

models = sf.models
fitted_model = sf.models[0]
fitted_model.arima_




# Check if the fitted model has the .arima_ attribute
if hasattr(fitted_model, "arima_"):
    arima_result = fitted_model.arima_
    # Extract coefficients as a pandas Series
    coefs = arima_result.params
    print(coefs)
else:
    print("No fitted ARIMA model found in fitted_model.")



# Find Best Model
df_best_model, df_cv_stat = best_model(df = df_trn, sf = sf,  h = h, n_windows = n_windows, step_size = step_size)


# Forecast Best Model 
df_fit, df_fct = forecast_best(df_trn = df_trn, df_best_model = df_best_model, model_dict = model_dict, h = h, freq = freq)

# Reconcile 
#df_rec = reconcile_best(df_fct = df_fct, df_fit = df_fit, S_df = S_df, tags = tags)
#df_rec = df_fct

# Mape Best Model
mape_value =MAPE(df_tst['y'].values, df_fct['yhat'].values)




#mape_best(df_fct = df_fct, df_tst = df_tst)

# Plot 
plot_result(df = df_all, df_fct = df_fct , df_tst = df_tst, n_tail = 36, total = True)



