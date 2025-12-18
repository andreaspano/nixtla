import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Carico il dataset
data = load_diabetes()
X = data.data       # features
y = data.target     # target (valore continuo)

print("Shape X:", X.shape)
print("Shape y:", y.shape)

# 2. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Standardizzazione (molto importante per Lasso)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 4. Modello Lasso con alpha fissato
lasso = Lasso(alpha=0.1, random_state=42)
lasso.fit(X_train_scaled, y_train)

y_pred = lasso.predict(X_test_scaled)

print("\n=== Lasso (alpha=0.1) ===")
print("R^2 test:", r2_score(y_test, y_pred))
print("MAE test:", mean_absolute_error(y_test, y_pred))

print("Coefficienti Lasso:")
for name, coef in zip(data.feature_names, lasso.coef_):
    print(f"{name:10s} -> {coef:.4f}")

# 5. LassoCV: selezione automatica di alpha con cross-validation
alphas = np.logspace(-3, 1, 50)  # da 0.001 a 10
lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)

y_pred_cv = lasso_cv.predict(X_test_scaled)

print("\n=== LassoCV (alpha ottimale) ===")
print("Alpha scelto:", lasso_cv.alpha_)
print("R^2 test:", r2_score(y_test, y_pred_cv))
print("MAE test:", mean_absolute_error(y_test, y_pred_cv))

print("Coefficienti LassoCV:")
for name, coef in zip(data.feature_names, lasso_cv.coef_):
    print(f"{name:10s} -> {coef:.4f}")

#################################
# Grid search example
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline

# 1. Dataset
data = load_diabetes()
X = data.data
y = data.target
feature_names = data.feature_names

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Pipeline: StandardScaler + Lasso
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", Lasso(random_state=42, max_iter=10000))
])

# 4. GridSearchCV per trovare alpha
param_grid = {
    "lasso__alpha": np.logspace(-3, 1, 20)
}

grid = GridSearchCV(
    pipe, param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid.fit(X_train, y_train)

# 5. Estraggo scaler e modello Lasso dalla pipeline
scaler = grid.best_estimator_.named_steps["scaler"]
lasso  = grid.best_estimator_.named_steps["lasso"]

# 6. Coefficienti scalati (nella metrica standardizzata)
coef_scaled = lasso.coef_
intercept_scaled = lasso.intercept_

# 7. Inversione della standardizzazione
coef_original = coef_scaled / scaler.scale_
intercept_original = intercept_scaled - np.sum(coef_original * scaler.mean_)

# 8. Output
print("=== MIGLIOR ALPHA ===")
print(grid.best_params_["lasso__alpha"])
print()

print("=== COEFFICIENTI SCALATI (STANDARDIZZATI) ===")
for name, c in zip(feature_names, coef_scaled):
    print(f"{name:10s} -> {c:.4f}")
print()

print("=== COEFFICIENTI NELLA SCALA ORIGINALE ===")
for name, c in zip(feature_names, coef_original):
    print(f"{name:10s} -> {c:.4f}")

print("\nIntercetta (scala originale):", intercept_original)
