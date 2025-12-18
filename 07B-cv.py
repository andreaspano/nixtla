from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, HistoricAverage
from statsforecast.utils import AirPassengersDF

from defs07B import *

#Data
df = AirPassengersDF

# Models
M0 = HistoricAverage(alias = 'Mean')
M1 = AutoARIMA(season_length=12, alias = 'Arima') #Forecasting model 
M2 = AutoETS(season_length=12, alias = 'ETS')

# Parameters
m = 12 #A minimum training size
h = 12 #forecast horizon: h
s = 1 #step size: s
n = 12 #number of windows: n
freq = 'ME' #frequency of data

# Instantiate StatsForecast class as sf
sf = StatsForecast(
    models=[M0, M1, M2],
    freq=freq
)


# Cross-validation 
df_cv = sf.cross_validation(
        df=df,
        h=h,
        step_size=s,
        n_windows=n,
        time_col = 'ds',
        target_col = 'y',
        id_col = 'unique_id'
    )

# Calculate Mape
df_mape = mape (df_cv, models = ['Mean', 'Arima', 'ETS'], id_col = 'unique_id', target_col = 'y')
       
#----------------------------------#
#      More analyses on df_cv      #
#----------------------------------#

# Restructure df_cv
df_cv = restruct(df_cv)

# Plot within cv window
plt_wth (df_cv)

# Plot between cv windows
plt_btw(df_cv, h = h) 


