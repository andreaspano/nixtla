import numpy as np

from siuba import group_by, summarize, filter,mutate, ungroup, arrange,  _
from siuba.dply.vector import row_number, dense_rank
from siuba.experimental.pivot import  pivot_longer
from utilsforecast.losses import mape
#from plotnine import ggplot, aes, geom_line, facet_wrap, geom_density , geom_histogram, geom_point, geom_linerange, element_blank, element_text, theme,  theme_bw, scale_x_continuous, xlab, ylab
from plotnine import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm



def restruct(df_cv): 

    # transform cutoff from dates to integer ranks
    df_cv = (df_cv
        >> arrange(_.cutoff)
        >> mutate(cutoff = dense_rank(_.cutoff))
    )


    # Long Format 
    df_cv = (
        df_cv
        >> pivot_longer(
            -_.unique_id, -_.ds, -_.cutoff, -_.y,
            names_to="model",
            values_to="yhat"
        )
        >> mutate(ape = np.abs((_.y - _.yhat) / _.y)) 
    )

    # add forecast horizon h
    df_cv  = (
        df_cv 
        >> group_by(_.unique_id, _.model, _.cutoff) 
        >> mutate(h = row_number(_.ds))
        >> ungroup()
    )

    return df_cv


# Plot within cv window
def plt_btw (df_cv, h): 


    # df for plotting h 
    df_plt_btw = (df_cv 
        >> group_by(_.model, _.h)
        >> summarize(avg_ape = np.mean(_.ape), std_ape = np.std(_.ape), include_groups = True) 
        >> mutate( min_ape = _.avg_ape - 2*_.std_ape, max_ape = _.avg_ape + 2*_.std_ape)
        
    )

    # Plot APE by model and horizon
    plt_btw = (
        ggplot(df_plt_btw, aes(x='h', y='avg_ape', color='model')) 
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

    return plt_btw
    