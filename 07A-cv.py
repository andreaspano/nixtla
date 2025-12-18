import numpy as np

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, HistoricAverage
from statsforecast.utils import AirPassengersDF
from siuba import group_by, summarize, filter,mutate, ungroup, arrange,  _

from siuba.dply.vector import row_number, dense_rank

from siuba.experimental.pivot import  pivot_longer
from utilsforecast.losses import mape
#from plotnine import ggplot, aes, geom_line, facet_wrap, geom_density , geom_histogram, geom_point, geom_linerange, element_blank, element_text, theme,  theme_bw, scale_x_continuous, xlab, ylab
from plotnine import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm



#Data
df = AirPassengersDF

# Models
M0 = HistoricAverage(alias = 'Mean')
M1 = AutoARIMA(season_length=12, alias = 'Arima') #Forecasting model 
M2 = AutoETS(season_length=12, alias = 'ETS')
M = [M0, M1, M2]

m = 12 #A minimum training size
h = 12 #forecast horizon: h
s = 1 #step size: s
n = 12 #number of windows: n
freq = 'ME' #frequency of data

sf = StatsForecast(
    models=[M0, M1, M2],
    freq=freq
)

#A starting point o: usually the origin
#A loss function: L(y, ŷ) (e.g., RMSE, MAE, MAPE)



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




df_cv = (df_cv
    >> arrange(_.cutoff)
    >> mutate(cutoff = dense_rank(_.cutoff))
)


# 

df_cv = (
    df_cv
    >> pivot_longer(
        -_.unique_id, -_.ds, -_.cutoff, -_.y,
        names_to="model",
        values_to="yhat"
    )
    >> mutate(ape = np.abs((_.y - _.yhat) / _.y)) 
)


df_cv  = (
    df_cv 
    >> group_by(_.unique_id, _.model, _.cutoff) 
    >> mutate(h = row_number(_.ds))
    >> ungroup()
)


mape_overall = (
df_cv 
    >> group_by(_.model)
    >> summarize(overall_mape = np.mean(_.ape), include_groups = True)
)



# df for plotting h 
df_plot_h = (df_cv 
    >> group_by(_.model, _.h)
    >> summarize(avg_ape = np.mean(_.ape), std_ape = np.std(_.ape), include_groups = True) 
    >> mutate( min_ape = _.avg_ape - 2*_.std_ape, max_ape = _.avg_ape + 2*_.std_ape)
    
)

# Plot APE by model and horizon
pl1 = (
    ggplot(df_plot_h, aes(x='h', y='avg_ape', color='model')) 
        + geom_point()  
        + geom_line() 
        + geom_linerange(aes( ymin='min_ape', ymax='max_ape'))
        + scale_x_continuous(breaks = np.arange(1, h+1))
        + facet_wrap('~model') 
        + theme_bw() 
        + xlab('Forecast Horizon (h)') 
        + ylab ('APE with 95% CI')
        + theme(
            legend_position = "bottom",
            legend_text = element_text(size = 12),
            legend_title = element_blank(),
            axis_title_x = element_text(size = 16)
        )
)



# df for plotting h 
df_plot_cutoff = (df_cv 
    >> group_by(_.model, _.cutoff)
    >> summarize(avg_ape = np.mean(_.ape), std_ape = np.std(_.ape), include_groups = True) 
    >> mutate( min_ape = _.avg_ape - 2*_.std_ape, max_ape = _.avg_ape + 2*_.std_ape)
    
)

# Plot APE by model and horizon
pl2 = (
    ggplot(df_plot_cutoff, aes(x='cutoff', y='avg_ape', color='model')) 
        + geom_point()  
        + geom_line() 
        + geom_linerange(aes( ymin='min_ape', ymax='max_ape'))
        + scale_x_continuous(breaks = np.arange(1, n+1))
        + facet_wrap('~model')
        + xlab('Test Window') 
        + ylab ('MAPE with 95% CI')
        + theme_bw() 
        + theme(
            #panel_grid = element_blank(),
            legend_position = "bottom",
            legend_text = element_text(size = 12),
            legend_title = element_blank(),
            axis_title = element_text(size = 16)
        )
)


ggsave("pl2.png", plot = pl2, width = 8, height = 6)  # adjust height as needed

# ANOVA table
#anova_table = anova_lm(model, typ=2)   # typ=2 = same logic as R's aov()
#print(anova_table)


## fit linear model
#model = smf.ols('ape ~ C(model) / C(model):C(h)', data=df_cv).fit()
#anova_table = anova_lm(model, typ=2)
#print(anova_table)
