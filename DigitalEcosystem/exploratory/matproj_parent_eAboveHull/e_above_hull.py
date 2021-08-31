#!/usr/bin/env python
# coding: utf-8

# # Filtering by E above hull
# 
# In this notebook, we train an XGBoost regressor to predict exfoliation energies. Data are filtered based on the stability of the parent structur in materials project.

# In[]:


import copy
import os
import collections

import pandas as pd
import numpy as np
import optuna
import xgboost
import imblearn.over_sampling
import sklearn.model_selection
import dscribe.descriptors
import tqdm
import sklearn.pipeline
import pymatgen.ext.matproj
import functools

import matplotlib.pyplot as plt
import sklearn.impute
import seaborn as sns

import sys
sys.path.append("../../../")
import DigitalEcosystem.utils.figures
import DigitalEcosystem.utils.misc
import DigitalEcosystem.utils.functional
from DigitalEcosystem.utils.misc import noble_gases, fblock, d_synths, matminer_descriptors

tqdm.tqdm.pandas()


# # Read in the Dataset

# In[]:


# Random seeds for reproducibility
RANDOM_SEED = 1234
import random
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# In[]:


# Load up the data
data_path = "../../refined/httpot/full_featurized_data.pkl"
data = pd.read_pickle(data_path)

data = data[data['discovery_process (unitless)']=='top-down']


# # Filter the Dataset
# 
# The next few cells look up the e above hull of the parent system from materials project.

# In[]:


data['mp_id (unitless)'] = data['2dm_id (unitless)'].progress_apply(DigitalEcosystem.utils.misc.get_parent_structure_id)
data['mp_id (unitless)']


# In[]:


rester = pymatgen.ext.matproj.MPRester(api_key=os.getenv("MATERIALS_PROJECT_API_KEY"))

@DigitalEcosystem.utils.functional.except_with_default_value()
def e_above_hull_fun(matproj_id):
    return DigitalEcosystem.utils.misc.get_e_above_hull(matproj_id, pymatgen_rester=rester)

df = data[data['mp_id (unitless)'] != 'no_parent']
df['e_above_hull'] = df['mp_id (unitless)'].progress_apply(e_above_hull_fun)


# In[]:


df.to_csv("data_with_mp_ids.csv")


# # Show the distribution of the data
# 
# In the below plot, we see a Pareto-esque distribution of the data.

# In[]:


plt.rcParams['figure.figsize'] = [20,10]
plt.rcParams['font.size'] = 24
sns.histplot(df['e_above_hull'])


# # Filter out NA entries in the dataset

# In[]:


df = df[df['e_above_hull'].isna() == False]


# In[]:


df = df.fillna(0)


# # Descriptor selection
# 
# XenonPy and Matminer

# In[]:


xenonpy_descriptors = [col for col in data.columns if ":" in col]

xenonpy_matminer_descriptors = xenonpy_descriptors + matminer_descriptors
target = ['exfoliation_energy_per_atom (eV/atom)']


# # Filter out by several masks
# 
# - `element_mask` - throw away systems containing noble gases, f-blocks, or any synthetic elements
# - `decomposition_mask` - keep systems with a decomposition energy < 0.5 eV/atom
# - `exfol_mask` - keep systems with an exfoliation energy > 0 eV/atom
# - `hull_mask` - keep systems whose parent structures e_above_hull reported by Materials Project is <= 0.05
# 
# And finally, do a train/test split

# In[]:


bad_elements = noble_gases + fblock + d_synths

element_mask = df['atoms_object (unitless)'].apply(lambda atoms: all([forbidden not in atoms.get_chemical_symbols() for forbidden in bad_elements]))

decomposition_mask = df['decomposition_energy (eV/atom)'] < 0.5

exfol_mask = df['exfoliation_energy_per_atom (eV/atom)'] > 0

hull_mask = df['e_above_hull'] <= 0.05

reasonable = df[element_mask & decomposition_mask & exfol_mask & hull_mask]

train, test = sklearn.model_selection.train_test_split(reasonable, test_size=0.1, random_state=RANDOM_SEED)


# In[]:


train_x_regression = np.nan_to_num(train[xenonpy_matminer_descriptors + ['e_above_hull']].to_numpy())
train_y_regression = np.nan_to_num(train[target].to_numpy())

test_x_regression = np.nan_to_num(test[xenonpy_matminer_descriptors + ['e_above_hull']].to_numpy())
test_y_regression = np.nan_to_num(test[target].to_numpy())


# # XGBoost Hyperparameter Tuning
# 
# Tune an XGBoost regressor for the exfoliation energy using Optuna.

# In[]:


current_regression = None
best_regression = None
def keep_best_regression(study, trial):
    global best_regression
    if study.best_trial == trial:
        best_regression = current_regression

objective_train_x_regression, objective_validation_x_regression, objective_train_y_regression, objective_validation_y_regression = sklearn.model_selection.train_test_split(
    np.nan_to_num(train_x_regression), train_y_regression, test_size=0.1, random_state=RANDOM_SEED)

def objective(trial: optuna.Trial):
    global current_regression

    params = {

    }

    current_regression = sklearn.pipeline.Pipeline([
        ("Scaler", sklearn.preprocessing.MinMaxScaler()),
        ("XGB_Regressor", xgboost.sklearn.XGBRegressor(
            max_depth= trial.suggest_int('max_depth', 1, 100),
            min_child_weight= trial.suggest_float('min_child_weight', 0, 100),
            reg_alpha = trial.suggest_float('alpha', 0, 5),
            reg_lambda = trial.suggest_float('lambda', 0, 5),
            n_estimators=200,
            objective='reg:squarederror',
            random_state=RANDOM_SEED),),
    ])

    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, f'validation_0-rmse')
    current_regression.fit(X=objective_train_x_regression, y=objective_train_y_regression,
                         **{
                            'XGB_Regressor__eval_set': [[objective_validation_x_regression, objective_validation_y_regression]],
                            'XGB_Regressor__eval_metric': 'rmse',
                            'XGB_Regressor__early_stopping_rounds': 20,
                            'XGB_Regressor__callbacks': [pruning_callback],
                            'XGB_Regressor__verbose': False
                         })

    score = sklearn.metrics.mean_squared_error(
        y_true=objective_validation_y_regression,
        y_pred=abs(current_regression.predict(objective_validation_x_regression)),
    )

    return np.sqrt(score)

regression_study = optuna.create_study(
    sampler = optuna.samplers.TPESampler(
        seed = RANDOM_SEED,
        warn_independent_sampling = True,
        consider_endpoints = True
    ),
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=1,
        max_resource=200),
    direction='minimize'
)

regression_study.optimize(func=objective, n_trials=1024, callbacks=[keep_best_regression])


# # Save summary statistics
# 
# - A parity plot for the model and the entire data range
# - A bar chart for some of the variable importances
# - Model performance statistics are also printed
# 

# In[]:


DigitalEcosystem.utils.figures.save_parity_plot(train_x_regression,
                                                test_x_regression,
                                                train_y_regression,
                                                test_y_regression,
                                                best_regression,
                                                "Exfoliation_energy (eV/atom)",
                                                "parity.jpeg")


# In[]:


n_importances = 20
importances = list(zip(best_regression[1].feature_importances_, xenonpy_matminer_descriptors))

sorted_importances = list(sorted(importances, key=lambda i: -i[0]))



plt.barh(range(n_importances), [imp[0] for imp in sorted_importances[:n_importances]])
plt.yticks(range(n_importances), [imp[1] for imp in sorted_importances[:n_importances]])
plt.ylabel("Feature")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("Importances.jpeg")


# In[]:


def rmse(y_true, y_pred):
    mse = sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)
    rmse = np.sqrt(abs(mse))
    return rmse

metrics = {
    'MaxError': sklearn.metrics.max_error,
    'MAE': sklearn.metrics.mean_absolute_error,
    'MSE': sklearn.metrics.mean_squared_error,
    'RMSE': rmse,
    'MAPE': sklearn.metrics.mean_absolute_percentage_error,
    'R2': sklearn.metrics.r2_score
}

y_pred_test = best_regression.predict(test_x_regression)
print("Test-Set Error Metrics")
for key, fun in metrics.items():
    value = fun(y_true=test_y_regression, y_pred=y_pred_test)
    print(key,np.round(value,3))


# In[ ]:




