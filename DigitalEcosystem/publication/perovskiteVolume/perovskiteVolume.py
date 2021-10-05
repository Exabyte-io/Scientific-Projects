#!/usr/bin/env python
# coding: utf-8

# # Perovskites
# 
# In this notebook, we demonstrate some of the initial work on this project that was carried out in the summer of 2021. Here, we leverage some Xenonpy-calculated properties to calculate the unit cell volume of several perovskite systems taken from NOMAD's database.
# 
# Both SISSO and TPOT are used to train predictive models, and near the end of the document we discuss what we can learn from this comparison.
# 
# Overall, in this notebook we demonstrate SISSO's ability to generate simple, interpretable models that can help lead us to physical insight - one of the major strengths of symbolic regression.

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

import xenonpy.descriptor

import sys, os

sys.path.append("../../../")
import DigitalEcosystem.utils.figures

from IPython.display import Latex

pd.options.mode.chained_assignment = None 


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
# Read the data, and featurize using XenonPy

# In[]:


data = pd.read_pickle('../../raw_data/perovskites.pkl')

# Scale volume to have units of Å^3 / formula unit
data["Volume"] /= data["Atoms_Object"].apply(lambda atoms: len(atoms)//5)

# Featurize with XenonPy
cal = xenonpy.descriptor.Compositions()
data["Symbols"] = data.Atoms_Object.apply(lambda atoms: collections.Counter(atoms.get_chemical_symbols()))
featurized_data = pd.concat([data, cal.transform(data.Symbols)], axis=1)

data = featurized_data.drop(columns=['Symbols'])
data


# # Prepare Data

# In[]:


target_column = ['Volume']
xenonpy_descriptors = [col for col in data.columns if ":" in col]

descriptors = xenonpy_descriptors

train, test = sklearn.model_selection.train_test_split(data, test_size=0.1, random_state=RANDOM_SEED)

train_x = train[descriptors].to_numpy()
train_y = train[target_column].to_numpy()

test_x = test[descriptors].to_numpy()
test_y = test[target_column].to_numpy()


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
                                                       axis_label = "Perovskite Volume (Å^3 / formula unit)",
                                                       filename = "xgboost_perovskite_volume_parity.jpeg")


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
plt.yticksRANDOM_SEEDnge(n_importances), [imp[1] for imp in sorted_importances[:n_importances]])
plt.ylabel("Feature")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("xgboost_importances.jpeg")


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

DigitalEcosystem.utils.figures.publication_parity_plot(train_y_true = train_y,
                                                       train_y_pred = tpot_model.predict(train_x),
                                                       test_y_true = test_y,
                                                       test_y_pred = tpot_model.predict(test_x),
                                                       axis_label = "Perovskite Volume (Å^3 / formula unit)",
                                                       filename = "tpot_perovskite_volume_parity.jpeg")


# In[]:


print("Test Set Error Metrics")
for key, fun in metrics.items():
    value = fun(y_true=test_y, y_pred=tpot_model.predict(test_x))
    print(key,np.round(value,4))
    
print("\nTraining Set Error Metrics")
for key, fun in metrics.items():
    value = fun(y_true=train_y, y_pred=tpot_model.predict(train_x))
    print(key,np.round(value,4))


# # Roost

# In[]:


roost_dir = "./roost"
os.makedirs(roost_dir, exist_ok=True)

roost_data_train = train[['Formula', 'Volume']]
roost_data_test = test[['Formula', 'Volume']]

roost_data_train.to_csv(os.path.join(roost_dir, 'roost_train.csv'), index_label='material_id')
roost_data_test.to_csv(os.path.join(roost_dir, 'roost_test.csv'), index_label='material_id')


# At this point, Roost models were run. Logs can be found in the Roost directory, along with the resultant predictions.

# In[]:


roost_train_results = pd.read_csv("roost/roost_train_predictions.csv", index_col="material_id")
roost_test_results  = pd.read_csv("roost/roost_test_predictions.csv", index_col="material_id")


# In[]:


DigitalEcosystem.utils.figures.publication_parity_plot(train_y_true = roost_train_results['volume_target'],
                                                       train_y_pred =  roost_train_results['volume_pred_n0'],
                                                       test_y_true = roost_test_results['volume_target'],
                                                       test_y_pred = roost_test_results['volume_pred_n0'],
                                                       axis_label = "Perovskite Volume (Å^3 / formula unit)",
                                                       filename = "roost_perovskite_volume_parity.jpeg")


# In[]:


print("Test Set Error Metrics")
for key, fun in metrics.items():
    value = fun(y_true=roost_test_results['volume_target'], y_pred=roost_test_results['volume_pred_n0'])
    print(key,np.round(value,4))
    
print("\nTraining Set Error Metrics")
for key, fun in metrics.items():
    value = fun(y_true=roost_train_results['volume_target'], y_pred=roost_train_results['volume_pred_n0'])
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

sisso_data_train.to_csv(os.path.join(sisso_dir, 'sisso_train.csv'), index_label='material_id')


# At this point, a SISSO model was run.

# In[]:


sisso_models = {
    'r1_1term': lambda df: -4.559505163148324e+02 + \
                           2.464859419692384e+00 * (df['ave:atomic_volume'] + df['ave:atomic_radius_rahm']),

    'r1_2term': lambda df: 7.027013257763095e+01 + \
                           -7.832363248251127e-01 * (df['ave:bulk_modulus'] - df['ave:atomic_number']) + \
                           6.564150096290632e-13 * (df['ave:atomic_radius_rahm'] ** 6),
    
    'r1_3term': lambda df: -3.526598287867814e+02 + \
                           -5.798450768280212e-02 * (df['var:c6_gb'] / df['sum:hhi_r']) + \
                           -5.504850208135466e-01 * (np.cbrt(df['var:melting_point'])) + \
                           2.201736063422449e+00 * (df['ave:atomic_volume'] + df['ave:atomic_radius_rahm']),
    
    'r1_4term': lambda df: -2.747502839404784e+02 + \
                           -5.185525800282890e-02 * (df['var:c6_gb'] / df['sum:hhi_p']) + \
                           -7.290004701356699e-19 * (df['ave:boiling_point'] ** 6) + \
                           -4.455507677046804e-01 * (df['ave:bulk_modulus'] - df['ave:atomic_weight']) + \
                           1.699729115030021e+00 * (df['ave:atomic_volume'] + df['ave:atomic_radius_rahm']),
    
    'r2_1term': lambda df: -2.729296923117930e+01 + \
                           -3.137476417692739e-03 * ((df['ave:bulk_modulus'] - df['ave:atomic_radius_rahm']) * (df['ave:atomic_weight'] + df['ave:atomic_radius_rahm'])),
    
    'r2_2term': lambda df: -2.063184570690620e+01 + \
                           -1.686398694159229e+04 * ((df['var:c6_gb'] / df['ave:bulk_modulus']) / (df['ave:atomic_volume'] ** 6)) + \
                           -3.195005086243114e-03 * ((df['ave:bulk_modulus'] - df['ave:atomic_radius_rahm']) * (df['ave:atomic_weight'] + df['ave:atomic_radius_rahm'])),
    
    'r2_3term': lambda df: -1.510330342353259e+01 + \
                           4.058647378703507e+03 * ((df['var:sound_velocity'] - df['var:hhi_p']) / (df['var:c6_gb'] * df['sum:hhi_r'])) + \
                           -2.139489205523803e+04 * ((df['var:c6_gb'] / df['ave:bulk_modulus']) / df['ave:atomic_volume'] ** 6) + \
                           -3.170219793913757e-03 * ((df['ave:bulk_modulus'] - df['ave:atomic_radius_rahm']) * (df['ave:atomic_weight'] + df['ave:atomic_radius_rahm'])),
    
    'r2_4term': lambda df: -1.199519222577036e+01 + \
                           1.884199571337176e+00 * ((df['ave:boiling_point'] * df['ave:atomic_weight']) / (abs(df['var:melting_point'] - df['sum:hhi_r']))) + \
                           3.941803378766428e+03 * ((df['var:sound_velocity'] - df['var:hhi_p']) / (df['var:c6_gb'] * df['sum:hhi_r'])) + \
                           -2.184309236205812e+04 * ((df['var:c6_gb'] / df['ave:bulk_modulus']) / (df['ave:atomic_volume'] ** 6)) + \
                           -3.048156404473468e-03 * ((df['ave:bulk_modulus'] - df['ave:atomic_radius_rahm']) * (df['ave:atomic_weight'] + df['ave:atomic_radius_rahm']))
}

for key, fun in sisso_models.items():
    print(f"==========\nSISSO Model {key}")
    sisso_train_predictions = fun(sisso_data_train)
    sisso_test_predictions = fun(sisso_data_test)
    sisso_data_train[key] = sisso_train_predictions
    sisso_data_test[key] = sisso_test_predictions
    
    print("\nTest Set Error Metrics")
    for metric, fun in metrics.items():
        value = fun(y_true=sisso_data_test['Volume'], y_pred=sisso_test_predictions)
        print(metric,np.round(value,4))

    print("\nTraining Set Error Metrics")
    for metric, fun in metrics.items():
        value = fun(y_true=sisso_data_train['Volume'], y_pred=sisso_train_predictions)
        print(metric,np.round(value,4))
    
    


# In[]:


sisso_data_train


# It's not the best model, but we're gonna use the Rung1 1Term model, because it's simple and still performs well. It's also intuitive.

# In[]:


sisso_data_train.to_csv(os.path.join(sisso_dir, 'sisso_results_train.csv'))
sisso_data_test.to_csv(os.path.join(sisso_dir, 'sisso_results_test.csv'))


# In[]:


model_to_plot = 'r1_1term'
DigitalEcosystem.utils.figures.publication_parity_plot(train_y_true = sisso_data_train['Volume'],
                                                       train_y_pred = sisso_data_train[model_to_plot],
                                                       test_y_true = sisso_data_test['Volume'],
                                                       test_y_pred = sisso_data_test[model_to_plot],
                                                       axis_label = "Perovskite Volume (Å^3 / formula unit)",
                                                       filename = "sisso_perovskite_volume_parity.jpeg")


# In[ ]:




