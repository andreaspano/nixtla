
#################################
# Grid search example
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline


def select_x(y, X, cv, max_iter, random_state, scoring):

    feature_names = X.columns

    # Manual scaling before GridSearchCV (no pipeline)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    param_grid = {"alpha": np.logspace(-3, 3, 100)}

    grid = GridSearchCV(
        Lasso(random_state=random_state, max_iter=max_iter),
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    grid.fit(X_scaled, y)



    # Extract best Lasso model
    lasso = grid.best_estimator_
    coef = [name for name, coef in zip(feature_names, lasso.coef_) if coef != 0]


    return coef

##############################################

# Load Data
data = pd.read_csv('~/dev/ml/data/mtcars.csv')
# Target
y = data.mpg
# Features
X = data.drop(columns=['mpg', 'model'])


select_x(y=y, X=X, cv=5, max_iter=1000, random_state=46, scoring='r2')
