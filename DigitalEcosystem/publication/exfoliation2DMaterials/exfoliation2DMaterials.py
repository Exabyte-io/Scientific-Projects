#!/usr/bin/env python
# coding: utf-8

# # 2D Material Exfoliation
# 
# Plots for the exfoliation energy problem

# In[]:


import functools
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tpot
import sklearn
import optuna
import xgboost
import pymatgen
import xenonpy.descriptor
from tqdm.notebook import tqdm 
import sys, os

sys.path.append("../../../")
import DigitalEcosystem.utils.figures
from DigitalEcosystem.utils.functional import except_with_default_value
from DigitalEcosystem.utils.misc import matminer_descriptors
from DigitalEcosystem.utils.element_symbols import noble_gases, f_block_elements, synthetic_elements_in_d_block

from IPython.display import Latex

pd.options.mode.chained_assignment = None 
tqdm.pandas()


# In[]:


# Random seeds for reproducibility
RANDOM_SEED = 42
import random
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# In[]:


# Plot Configuration
plt.rcParams["figure.figsize"] = (15, 15)
plt.rcParams["font.size"] = 32


# # Read in the Data
# 
# Read the data

# In[]:


data = pd.read_pickle('../feature_engineering/full_featurized_data.pkl')

# =========================
# Convert eV/atom to J/m^2
# =========================

# Conversion factors
J_per_eV = 1.60217733e-19
m_per_A = 1e-10

# eV/atom -> eV
data['exfoliation_energy (eV)'] = data['exfoliation_energy_per_atom (eV/atom)'] * data['atoms_object (unitless)'].apply(len)
# eV -> J
data['exfoliation_energy (J)'] = data['exfoliation_energy (eV)'] * J_per_eV

# Get surface area of unit cell (magnitude of the cross-product of the A/B directions). ASE uses the Angstrom as its base unit
data['surface_area (A^2)'] = data['atoms_object (unitless)'].apply(lambda atoms: atoms.get_cell()).apply(lambda cell: np.linalg.norm(np.cross(cell[0], cell[1])))

# A^2 -> m^2
data['surface_area (m^2)'] = data['surface_area (A^2)'] * (m_per_A**2)

# J / m^2
data['exfoliation_energy (J/m^2)'] = data['exfoliation_energy (J)'] / data['surface_area (m^2)']

# ==============
# Data filtering
# ==============
target_column = ["exfoliation_energy (J/m^2)"]

# Drop any missing entries (some exfoliation energies are undefined by 2DMatPedia)
data = data[data[target_column[0]].notna()]

# # Drop anything in the f-block, larger than U, and noble gases
bad_elements = noble_gases + f_block_elements + synthetic_elements_in_d_block
element_mask = data['atoms_object (unitless)'].apply(lambda atoms: all([forbidden not in atoms.get_chemical_symbols() for forbidden in bad_elements]))

# Drop anything that decomposes
decomposition_mask = data['decomposition_energy (eV/atom)'] < 0.5

# Drop things with non-strictly-positive exfoliation energies
exfol_mask = data['exfoliation_energy_per_atom (eV/atom)'] > 0

data = data[element_mask & decomposition_mask & exfol_mask]
data


# In[]:


xenonpy_descriptors = [col for col in data.columns if ":" in col]
descriptors = xenonpy_descriptors + matminer_descriptors


# # Prepare Data

# In[]:


train, test = sklearn.model_selection.train_test_split(data, test_size=0.1, random_state=RANDOM_SEED)

train_x = np.nan_to_num(train[descriptors].to_numpy())
train_y = np.nan_to_num(train[target_column].to_numpy())

test_x = np.nan_to_num(test[descriptors].to_numpy())
test_y = np.nan_to_num(test[target_column].to_numpy())


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


# # XGBoost

# In[]:


current_reg = None
best_reg = None
def keep_best_reg(study, trial):
    global best_reg
    if study.best_trial == trial:
        best_reg = current_reg

objective_train_x_reg, objective_validation_x_reg, objective_train_y_reg, objective_validation_y_reg = sklearn.model_selection.train_test_split(
    np.nan_to_num(train_x), train_y, test_size=0.2, random_state=RANDOM_SEED)

def objective(trial: optuna.Trial):
    global current_reg


    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0, 2),
        'min_split_loss': trial.suggest_float('min_split_loss', 0, 2),
        'max_depth': trial.suggest_int('max_depth', 1, 256),
        'min_child_weight': trial.suggest_float('min_child_weight', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2)
    }
    
    scalers = {
        "StandardScaler": sklearn.preprocessing.StandardScaler(),
        "MinMaxScaler": sklearn.preprocessing.MinMaxScaler()
    }

    scaler = trial.suggest_categorical('scaler', scalers.keys())

    current_reg = sklearn.pipeline.Pipeline([
        (scaler, scalers[scaler]),
        ("XGB_Regressor", xgboost.sklearn.XGBRegressor(**params,
                                               n_estimators=256,
                                               n_jobs=1,
                                               objective='reg:squarederror',
                                               random_state=RANDOM_SEED),)
    ])

    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, f'validation_0-rmse')
    current_reg.fit(X=objective_train_x_reg, y=objective_train_y_reg,
                         **{
                            'XGB_Regressor__eval_set': [[objective_validation_x_reg, objective_validation_y_reg]],
                            'XGB_Regressor__eval_metric': 'rmse',
                            'XGB_Regressor__early_stopping_rounds': 50,
                            'XGB_Regressor__callbacks': [pruning_callback],
                            'XGB_Regressor__verbose': False
                         })

    mse = sklearn.metrics.mean_squared_error(
        y_true=objective_validation_y_reg,
        y_pred=current_reg.predict(objective_validation_x_reg),
    )
    rmse = np.sqrt(mse)

    return rmse

reg_study = optuna.create_study(
    sampler = optuna.samplers.TPESampler(
        seed = RANDOM_SEED,
        warn_independent_sampling = True,
        consider_endpoints = True
    ),
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=1,
        max_resource=256),
    direction='minimize')

reg_study.optimize(func=objective, n_trials=1000, callbacks=[keep_best_reg])


# In[]:


DigitalEcosystem.utils.figures.publication_parity_plot(train_y_true = train_y,
                                                       train_y_pred = best_reg.predict(train_x),
                                                       test_y_true = test_y,
                                                       test_y_pred = best_reg.predict(test_x),
                                                       axis_label = "Exfoliation Energy (J/m^2)",
                                                       filename = "xgboost_2dm_exfoliation_parity.jpeg")


# In[]:


print("Test Set Error Metrics")
for key, fun in metrics.items():
    value = fun(y_true=test_y, y_pred=best_reg.predict(test_x))
    print(key,np.round(value,4))
    
print("\nTraining Set Error Metrics")
for key, fun in metrics.items():
    value = fun(y_true=train_y, y_pred=best_reg.predict(train_x))
    print(key,np.round(value,4))


# In[]:


n_importances = 10
importances = list(zip(best_reg[1].feature_importances_, xenonpy_descriptors))

sorted_importances = list(sorted(importances, key=lambda i: -i[0]))

plt.barh(range(n_importances), [imp[0] for imp in sorted_importances[:n_importances]])
plt.yticks(range(n_importances), [imp[1] for imp in sorted_importances[:n_importances]])
plt.ylabel("Feature")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("xgboost_importances.jpeg")


# In[]:


train_preds = train[target_column]
train_preds['TrainTest Status'] = ['Training Set'] * len(train_preds)
train_preds['Prediction'] = best_reg.predict(train_x)

test_preds = test[target_column]
test_preds['TrainTest Status'] = ['Test Set'] * len(test_preds)
test_preds['Prediction'] = best_reg.predict(test_x)

xgb_predictions = train_preds.append(test_preds)
xgb_predictions.to_csv("xgboost_2dm_exfoliation_predictions.csv")


# In[]:


with open("xgboost_2dm_exfoliation_importances.csv", "w") as outp:
    outp.write("Descriptor,XGB_Importance\n")
    for importance, descriptor in sorted_importances:
        outp.write(f"{descriptor},{importance}\n")


# # TPOT

# In[]:


tpot_model = tpot.TPOTRegressor(
    generations=10,
    population_size=100,
    max_eval_time_mins=10 / 60,
    cv=10,
    verbosity=2,
    scoring="neg_root_mean_squared_error",
    config_dict=tpot.config.regressor_config_dict,
    n_jobs=-1,
    random_state=RANDOM_SEED
)

tpot_model.fit(train_x, train_y.ravel())


# In[]:


DigitalEcosystem.utils.figures.publication_parity_plot(train_y_true = train_y,
                                                       train_y_pred = tpot_model.predict(train_x),
                                                       test_y_true = test_y,
                                                       test_y_pred = tpot_model.predict(test_x),
                                                       axis_label = "Exfoliation Energy (J/m^2)",
                                                       filename = "tpot_2dm_exfoliation_parity.jpeg")


# In[]:


print("Test Set Error Metrics")
for key, fun in metrics.items():
    value = fun(y_true=test_y, y_pred=tpot_model.predict(test_x))
    print(key,np.round(value,4))
    
print("\nTraining Set Error Metrics")
for key, fun in metrics.items():
    value = fun(y_true=train_y, y_pred=tpot_model.predict(train_x))
    print(key,np.round(value,4))


# In[ ]:


train_preds = train[target_column]
train_preds['TrainTest Status'] = ['Training Set'] * len(train_preds)
train_preds['Prediction'] = tpot_model.predict(train_x)

test_preds = test[target_column]
test_preds['TrainTest Status'] = ['Test Set'] * len(test_preds)
test_preds['Prediction'] = tpot_model.predict(test_x)

tpot_predictions = train_preds.append(test_preds)
tpot_predictions.to_csv("tpot_2dm_bandgap_predictions.csv")


# 

# # Roost

# In[]:


roost_dir = "./roost"
os.makedirs(roost_dir, exist_ok=True)
roost_data_train = train[['formula'] + target_column]
roost_data_test = test[['formula'] + target_column]

roost_data_train.to_csv(os.path.join(roost_dir, 'roost_train.csv'), index_label='material_id')
roost_data_test.to_csv(os.path.join(roost_dir, 'roost_test.csv'), index_label='material_id')


# At this point, Roost models were run. Logs can be found in the Roost directory, along with the resultant predictions.

# In[]:


roost_train_results = pd.read_csv("roost/roost_train_predictions.csv", index_col="material_id")
roost_test_results  = pd.read_csv("roost/roost_test_predictions.csv", index_col="material_id")


# In[]:


DigitalEcosystem.utils.figures.publication_parity_plot(train_y_true = roost_train_results['exfoliation_energy_target'],
                                                       train_y_pred =  roost_train_results['exfoliation_energy_pred_n0'],
                                                       test_y_true = roost_test_results['exfoliation_energy_target'],
                                                       test_y_pred = roost_test_results['exfoliation_energy_pred_n0'],
                                                       axis_label = "Exfoliation Energy (J/m^2)",
                                                       filename = "roost_2dm_exfoliation_parity.jpeg")


# In[]:


print("Test Set Error Metrics")
for key, fun in metrics.items():
    value = fun(y_true=roost_test_results['exfoliation_energy_target'], y_pred=roost_test_results['exfoliation_energy_pred_n0'])
    print(key,np.round(value,4))
    
print("\nTraining Set Error Metrics")
for key, fun in metrics.items():
    value = fun(y_true=roost_train_results['exfoliation_energy_target'], y_pred=roost_train_results['exfoliation_energy_pred_n0'])
    print(key,np.round(value,4))


# # SISSO
# 
# Start by obtaining importance scores from the XGBoost model

# In[]:


sisso_feature_selector = sklearn.feature_selection.SelectFromModel(sklearn.linear_model.LassoCV(random_state=RANDOM_SEED),
                                                                   threshold=-np.inf,
                                                                   max_features=16,
                                                                   prefit=False)
sisso_feature_selector.fit(train_x, train_y.ravel())

sisso_features = [col for (col, is_selected) in zip(train[descriptors].columns, sisso_feature_selector.get_support()) if is_selected]
print("\n".join(sisso_features))


# In[]:


sisso_dir = "./sisso"
os.makedirs(sisso_dir, exist_ok=True)

sisso_data_train = train[target_column + sisso_features]
sisso_data_test = test[target_column + sisso_features]

sisso_data_train.to_csv(os.path.join(sisso_dir, 'sisso_train.csv'), index_label='2dm_id (unitless)')


# At this point, a SISSO model was run.

# In[]:


sisso_models = {
    'r1_1term': lambda df: 1.159576043776266e+00 + \
                           -5.585482268035945e-02 * (df['ave:atomic_volume'] - df['ave:Polarizability']),
    
    'r1_2term': lambda df: -6.414477361338824e-01 + \
                           7.234395756089002e-01 * (df['ave:Polarizability'] / df['ave:atomic_volume']) + \
                           1.041717291506195e-01 * (df['ave:atomic_radius'] / df['ave:atomic_volume']),
    
    'r1_3term': lambda df: -6.109742010210223e-01 + \
                           -1.274929386543976e-03 * (np.sqrt(df['var:evaporation_heat'])) + \
                           7.420876074363660e-01 * (df['ave:Polarizability'] / df['ave:atomic_volume']) + \
                           1.186593176328466e-01 * (df['ave:atomic_radius'] / df['ave:atomic_volume']),
    
    'r1_4term': lambda df: -1.333025981566247e+00 + \
                           1.656127669129025e+02 * (1 / df['ave:atomic_radius_rahm']) + \
                           -1.295307555975112e-03 * (np.sqrt(df['var:evaporation_heat'])) + \
                           1.009576663923102e+00 * (df['ave:Polarizability'] / df['ave:atomic_volume']) + \
                           1.028373562950152e-01 * (df['ave:atomic_radius'] / df['ave:atomic_volume']),
    
    'r2_1term': lambda df: 3.547073341936840e-01 + \
                           1.018594219452304e+00 * ((df['ave:atomic_radius'] * df['ave:Polarizability']) / (df['ave:atomic_volume']**3)),
    
    'r2_2term': lambda df: 4.116570822745532e-01 + \
                           -1.285959176439781e-04 * ((np.cbrt(df['var:evaporation_heat']))*(df['ave:atomic_radius_rahm'] - df['ave:atomic_radius'])) + \
                           1.744234251683070e+01 * ((df['ave:Polarizability']) / (df['ave:atomic_volume']**2)),
    
    'r2_3term': lambda df: -1.101628086686219e-01 + \
                           -1.267342505583142e-04 * ((np.cbrt(df['var:evaporation_heat'])) * (df['ave:atomic_radius_rahm'] - df['ave:atomic_radius'])) + \
                           2.591117120632469e-24 * ((df['var:thermal_conductivity'] * df['var:hhi_r']) * (df['var:hhi_p'] * df['var:c6_gb'])) + \
                           1.038004615974527e+01 * ((df['ave:atomic_volume'] + df['ave:Polarizability']) / (df['ave:atomic_volume'] **2)),
    
    'r2_4term': lambda df: -1.295680108715010e-01 + \
                           3.665239225465616e-02 * ((df['sum:boiling_point'] * df['ave:Polarizability']) / (df['var:evaporation_heat'] + df['sum:hhi_r'])) + \
                           -1.075253530950549e-04 * ((np.cbrt(df['var:evaporation_heat'])) * (df['ave:atomic_radius_rahm'] - df['ave:atomic_radius'])) + \
                           2.485158141474120e-24 * ((df['var:thermal_conductivity'] * df['var:hhi_r']) * (df['var:hhi_p'] * df['var:c6_gb'])) + \
                           9.428416741220271e+00 * ((df['ave:atomic_volume'] + df['ave:Polarizability']) / (df['ave:atomic_volume']**2))
}

for key, fun in sisso_models.items():
    print(f"==========\nSISSO Model {key}")
    sisso_train_predictions = fun(sisso_data_train)
    sisso_test_predictions = fun(sisso_data_test)
    sisso_data_train[key] = sisso_train_predictions
    sisso_data_test[key] = sisso_test_predictions
    
    print("\nTest Set Error Metrics")
    for metric, fun in metrics.items():
        value = fun(y_true=sisso_data_test['exfoliation_energy (J/m^2)'], y_pred=sisso_test_predictions)
        print(metric,np.round(value,4))

    print("\nTraining Set Error Metrics")
    for metric, fun in metrics.items():
        value = fun(y_true=sisso_data_train['exfoliation_energy (J/m^2)'], y_pred=sisso_train_predictions)
        print(metric,np.round(value,4))
    
    


# In[]:


sisso_data_train.to_csv(os.path.join(sisso_dir, 'sisso_results_train.csv'))
sisso_data_test.to_csv(os.path.join(sisso_dir, 'sisso_results_test.csv'))


# In[]:


model_to_plot = 'r2_4term'
DigitalEcosystem.utils.figures.publication_parity_plot(train_y_true = sisso_data_train['exfoliation_energy (J/m^2)'],
                                                       train_y_pred = sisso_data_train[model_to_plot],
                                                       test_y_true = sisso_data_test['exfoliation_energy (J/m^2)'],
                                                       test_y_pred = sisso_data_test[model_to_plot],
                                                       axis_label = "Exfoliation Energy (J/m^2)",
                                                       filename = "sisso_2dm_exfoliation_parity.jpeg")


# In[ ]:




