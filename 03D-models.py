# Siple example with a unique sella series 59 or 57
from defs import *
from plotnine import ggplot, aes, geom_line

h = 3
n_windows = 10
step_size = 3
min_lenght = 12
freq = 'MS'

# Read data
dtype={'year': int, 'month': object, 'unique_id': object, 'y':float, 'desc': float}
df = pd.read_csv('~/dev/ml/data/fondi-economics.csv', dtype=dtype)



# Prepare data
df = prepare(df, min_length= min_lenght)

# select 10B4000001057 only
#df = df[df['series_id'] == '10B4000001057']
df = df[df['series_id'] == '10B4000001059']



# rename series_id unique_id
df = df.rename(columns={'series_id': 'unique_id'})


(
    ggplot(df)  # What data to use
        + aes(x="ds", y="y")  # What variable to use
        + geom_line() 
)



df_trn , df_tst = split(df, h = h)


'''
sf = StatsForecast(
    models=[AutoARIMA(season_length=12)],
    freq='MS',
)
'''

sf = StatsForecast(
    models=[AutoARIMA()],
    freq='MS',
)

sf.fit(df_trn)


sf.fitted_[0][0].model_['coef']





df_fct = sf.forecast(df = df_trn, h=h )
df_prd = sf.predict( h=h )


