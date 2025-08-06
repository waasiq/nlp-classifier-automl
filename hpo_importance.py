import pandas as pd
import shap
from xgboost import XGBRegressor
import numpy as np

# Example dataframe
df = pd.DataFrame([
    {'lr': 0.001, 'batch_size': 32, 'val_auc': 0.85},
    {'lr': 0.01, 'batch_size': 64, 'val_auc': 0.87},
    # ...
])
X = df[['lr', 'batch_size']]
y = df['val_auc']

X['log_lr'] = np.log10(X['lr'])
X = X.drop(columns=['lr'])


model = XGBRegressor()
model.fit(X, y)


explainer = shap.Explainer(model)
shap_values = explainer(X)

# Summary plot
shap.summary_plot(shap_values, X)
