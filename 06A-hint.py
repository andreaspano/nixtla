import matplotlib.pyplot as plt
import pandas as pd
from neuralforecast.losses.pytorch import GMM, sCRPS
from datasetsforecast.hierarchical import HierarchicalData
from neuralforecast.models import  NHITS, HINT
from neuralforecast import NeuralForecast

from siuba import _, filter, group_by, summarize

# Auxiliary sorting
def sort_df_hier(Y_df, S_df):
    # NeuralForecast core, sorts unique_id lexicographically
    # by default, this class matches S_df and Y_hat_df order.    
    Y_df.unique_id = Y_df.unique_id.astype('category')
    Y_df.unique_id = Y_df.unique_id.cat.set_categories(S_df.index)
    Y_df = Y_df.sort_values(by=['unique_id', 'ds'])
    return Y_df

# Load TourismSmall dataset
horizon = 12
Y_df, S_df, tags = HierarchicalData.load('./data', 'TourismLarge')
Y_df['ds'] = pd.to_datetime(Y_df['ds'])
Y_df = sort_df_hier(Y_df, S_df)
level = [80,90]


Y_df >> filter ( _.unique_id == 'TotalAll')
(Y_df
    >> group_by(_.unique_id)
    >> summarize(n = _.unique_id.count())
)



# Instantiate HINT
# BaseNetwork + Distribution + Reconciliation
nhits = NHITS(h=horizon,
              input_size=24,
              loss=GMM(n_components=10, level=level),
              max_steps=2000,
              early_stop_patience_steps=10,
              val_check_steps=50,
              scaler_type='robust',
              learning_rate=1e-3,
              valid_loss=sCRPS(level=level))

model = HINT(h=horizon, S=S_df.values,
             model=nhits,  reconciliation='BottomUp')

# Fit and Predict
nf = NeuralForecast(models=[model], freq='MS')
Y_hat_df = nf.cross_validation(df=Y_df, val_size=12, n_windows=1)
Y_hat_df = Y_hat_df.reset_index()