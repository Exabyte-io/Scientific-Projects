#!/usr/bin/env python
# coding: utf-8

# # Exfoliation Energy
# 
# In this notebook, we train an XGBoost regressor to predict exfoliation energies normalized by surface area, to J/M^2.

# In[]:


import copy

import pandas as pd
import numpy as np
import optuna
import xgboost
import imblearn.over_sampling
import sklearn.model_selection
import dscribe.descriptors
import tqdm
import sklearn.pipeline

import functools

import matplotlib.pyplot as plt
import sklearn.impute
import seaborn as sns

import sys
sys.path.append("../../../")
import DigitalEcosystem.utils.figures
from DigitalEcosystem.utils.misc import matminer_descriptors
from DigitalEcosystem.utils.element_symbols import noble_gases, f_block_elements, synthetic_elements_in_d_block

tqdm.tqdm.pandas()


# In[]:


# Random seeds for reproducibility
RANDOM_SEED = 1234
import random
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# # Read in the Dataset

# In[]:


# Load up the data
data_path = "../../refined/httpot/full_featurized_data.pkl"
data = pd.read_pickle(data_path)


# # Descriptor selection
# 
# XenonPy and Matminer

# In[]:


xenonpy_descriptors = [col for col in data.columns if ":" in col]
xenonpy_matminer_descriptors = xenonpy_descriptors + matminer_descriptors


# # Convert to J/M^2

# In[]:


# Conversion factors from http://laser.chem.olemiss.edu/~nhammer/constants.html

data['exfoliation_energy (eV)'] = data['exfoliation_energy_per_atom (eV/atom)'] * data['atoms_object (unitless)'].apply(len)

J_per_eV = 1.60217733e-19
data['exfoliation_energy (J)'] = data['exfoliation_energy (eV)'] * J_per_eV

data['surface_area (A^2)'] = data['atoms_object (unitless)'].apply(lambda atoms: atoms.get_cell()).apply(lambda cell: np.linalg.norm(np.cross(cell[0], cell[1])))
m_per_A = 1e-10
data['surface_area (m^2)'] = data['surface_area (A^2)'] * (m_per_A**2)

data['exfoliation_energy (J/m^2)'] = data['exfoliation_energy (J)'] / data['surface_area (m^2)']

target_column = 'exfoliation_energy (J/m^2)'


# # Filter out by several masks
# 
# - `element_mask` - throw away systems containing noble gases, f-blocks, or any synthetic elements
# - `decomposition_mask` - keep systems with a decomposition energy < 0.5 eV/atom
# - `exfol_mask` - keep systems with an exfoliation energy > 0 eV/atom
# 
# And finally, do a train/test split

# In[]:


bad_elements = noble_gases + f_block_elements + synthetic_elements_in_d_block

element_mask = data['atoms_object (unitless)'].apply(lambda atoms: all([forbidden not in atoms.get_chemical_symbols() for forbidden in bad_elements]))

decomp_mask = data['decomposition_energy (eV/atom)'] < 0.5

exfol_mask = data['exfoliation_energy_per_atom (eV/atom)'] > 0

reasonable = data[element_mask & decomp_mask & exfol_mask]


# In[]:


train, test = sklearn.model_selection.train_test_split(reasonable, test_size=0.1, random_state=RANDOM_SEED)


# In[]:


train_x_reg = np.nan_to_num(train[xenonpy_matminer_descriptors].to_numpy())
train_y_reg = np.nan_to_num(train[target_column].to_numpy())

test_x_reg = np.nan_to_num(test[xenonpy_matminer_descriptors].to_numpy())
test_y_reg = np.nan_to_num(test[target_column].to_numpy())


# # XGBoost Hyperparameter Tuning
# 
# Tune an XGBoost regressor for the exfoliation energy using Optuna.

# In[]:


current_reg = None
best_reg = None
def keep_best_reg(study, trial):
    global best_reg
    if study.best_trial == trial:
        best_reg = current_reg

def objective(trial: optuna.Trial):
    global current_reg
    
    SEED = trial.suggest_categorical('random', [42,1234,12345])
    objective_train_x_reg, objective_validation_x_reg, objective_train_y_reg, objective_validation_y_reg = sklearn.model_selection.train_test_split(
    np.nan_to_num(train_x_reg), train_y_reg, test_size=0.25, random_state=SEED)

    current_reg = sklearn.pipeline.Pipeline([
        ("Scaler", sklearn.preprocessing.MinMaxScaler()),
        ("XGB_Regressor", xgboost.sklearn.XGBRegressor(
            max_depth= trial.suggest_int('max_depth', 1, 50),
            min_child_weight= trial.suggest_float('min_child_weight', 0, 10),
            reg_alpha = trial.suggest_float('alpha', 0, 10),
            reg_lambda = trial.suggest_float('lambda', 0, 10),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0,1),
            subsample=trial.suggest_float('subsample', .1,1),
            learning_rate=trial.suggest_float('learning_rate', 0.001, 1),
            n_estimators=1000,
            objective='reg:pseudohubererror',
            random_state=SEED)),
    ])
    
#     pruning_callback = optuna.integration.XGBoostPruningCallback(trial, f'validation_0-rmse')
    current_reg.fit(X=objective_train_x_reg, y=objective_train_y_reg,
                         **{
#                             'XGB_Regressor__eval_set': [[objective_validation_x_reg, objective_validation_y_reg]],
#                             'XGB_Regressor__eval_metric': 'rmse',
#                             'XGB_Regressor__early_stopping_rounds': 100,
#                             'XGB_Regressor__callbacks': [pruning_callback],
                            'XGB_Regressor__verbose': False
                         })

    score = sklearn.metrics.r2_score(
        y_true=objective_validation_y_reg,
        y_pred=abs(current_reg.predict(objective_validation_x_reg)),
    )

    return score

reg_study = optuna.create_study(
    sampler = optuna.samplers.TPESampler(
        seed = RANDOM_SEED,
        warn_independent_sampling = True,
        consider_endpoints = True
    ),
#     pruner = optuna.pruners.HyperbandPruner(
#         min_resource=1,
#         max_resource=1000),
    direction='maximize'
)

reg_study.optimize(func=objective, n_trials=128, callbacks=[keep_best_reg])


# # Save summary statistics
# 
# - A parity plot for the model and the entire data range
#     - Also for a subset of the range
# - Model performance statistics are also printed
# 

# In[]:


DigitalEcosystem.utils.figures.save_parity_plot(train_x_reg,
                                                test_x_reg,
                                                train_y_reg,
                                                test_y_reg,
                                                best_reg,
                                                target_column,
                                                "exfoliation_joules_per_meter.jpeg")


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

y_pred_test = best_reg.predict(test_x_reg)
for key, fun in metrics.items():
    value = fun(y_true=test_y_reg, y_pred=y_pred_test)
    print(key,np.round(value,3))


# In[]:


# Zoom in on just exfoliation energies below 2 eV
cutoff=2
DigitalEcosystem.utils.figures.save_parity_plot(train_x_reg[train_y_reg<cutoff, :],
                                                test_x_reg[test_y_reg<cutoff, :],
                                                train_y_reg[train_y_reg<cutoff],
                                                test_y_reg[test_y_reg<cutoff],
                                                best_reg,
                                                target_column,
                                                "exfoliation_joules_per_meter_lessThan2.jpeg")


# In[ ]:




