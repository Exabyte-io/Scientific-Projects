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

import sys

sys.path.append("../../../")
import DigitalEcosystem.utils.figures

from IPython.display import Latex

pd.options.mode.chained_assignment = None 


# In[]:


# Random seeds for reproducibility
RANDOM_SEED = 1234
import random
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


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

train, test = sklearn.model_selection.train_test_split(data, test_size=0.1, random_state=RANDOM_SEED)

train_x = train[xenonpy_descriptors].to_numpy()
train_y = train[target_column].to_numpy()

test_x = test[xenonpy_descriptors].to_numpy()
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
        'learning_rate': trial.suggest_float('learning_rate', 0, 1),
        'min_split_loss': trial.suggest_float('min_split_loss', 0, 1),
        'max_depth': trial.suggest_int('max_depth', 1, 100),
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
                                               n_estimators=100,
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

    score = sklearn.metrics.mean_poisson_deviance(
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
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=1,
        max_resource=100),
    direction='minimize')

reg_study.optimize(func=objective, n_trials=1000, callbacks=[keep_best_reg])


# In[]:


DigitalEcosystem.utils.figures.save_parity_plot(train_x,
                                                test_x,
                                                train_y,
                                                test_y,
                                                best_reg,
                                                "Perovskite Volume (Å^3 / formula unit)",
                                                "xgboost_perovskite_volume_parity.jpeg")


# In[]:


print("Test Set Error Metrics")
for key, fun in metrics.items():
    value = fun(y_true=test_y, y_pred=best_reg.predict(test_x))
    print(key,np.round(value,4))
    
print("\nTraining Set Error Metrics")
for key, fun in metrics.items():
    value = fun(y_true=train_y, y_pred=best_reg.predict(train_x))
    print(key,np.round(value,4))


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
    random_state=1234
)

tpot_model.fit(train_x, train_y.ravel())


# In[]:


DigitalEcosystem.utils.figures.save_parity_plot(train_x,
                                                test_x,
                                                train_y,
                                                test_y,
                                                tpot_model,
                                                "Perovskite Volume (Å^3 / formula unit)",
                                                "tpot_perovskite_volume_parity.jpeg")


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


import os
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


plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
plt.rcParams["figure.figsize"] = (15, 15)
plt.rcParams["font.size"] = 16

plt.scatter(x=roost_train_results['volume_target'], y=roost_train_results['volume_pred_n0'], label="Train Set")
plt.scatter(x=roost_test_results['volume_target'], y=roost_test_results['volume_pred_n0'], label="Test Set")

min_xy = min(min(roost_train_results['volume_target']),
             min(roost_test_results['volume_target']),
             min(roost_train_results['volume_pred_n0']),
             min(roost_test_results['volume_pred_n0']))
max_xy = max(max(roost_train_results['volume_target']),
             max(roost_test_results['volume_target']),
             max(roost_train_results['volume_pred_n0']),
             max(roost_test_results['volume_pred_n0']))

plt.plot([min_xy, max_xy], [min_xy, max_xy], label="Parity")
plt.ylabel(f"Perovskite Volume (Å^3 / formula unit) (Predicted)")
plt.xlabel(f"Perovskite Volume (Å^3 / formula unit) (Dataset)")
plt.legend()
plt.savefig("roost_perovskite_volume_parity.jpeg")
plt.show()
plt.close()


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


important_features = [record[1] for record in sorted_importances[:n_importances]]

sisso_dir = "./sisso"
os.makedirs(sisso_dir, exist_ok=True)

sisso_data_train = train[target_column + important_features]
sisso_data_test = test[target_column + important_features]

sisso_data_train.to_csv(os.path.join(sisso_dir, 'sisso_train.csv'), index_label='material_id')


# At this point, a SISSO model was run.

# In[]:


sisso_models = {
    'r1_1term': lambda df: -1.173401497819689e+02 + \
                           1.710176880247700e-01 * (df['ave:vdw_radius_uff'] * df['ave:gs_est_bcc_latcnt']),
    
    'r1_2term': lambda df: -7.912354154958560e+01 + \
                           -1.609923938541456e-02 * (df['ave:bulk_modulus'] * df['ave:gs_volume_per']) + \
                           1.559313572968614e-01 * (df['ave:vdw_radius_uff'] * df['ave:gs_est_bcc_latcnt']),
    
    'r1_3term': lambda df: -6.567261626600551e+01 + \
                           -2.978198018981838e+01 * (df['ave:gs_volume_per'] / df['ave:atomic_weight']) + \
                           -1.704453039727385e-02 * (df['ave:bulk_modulus'] * df['ave:gs_volume_per']) + \
                           1.581938411940554e-01 * (df['ave:vdw_radius_uff'] * df['ave:gs_est_bcc_latcnt']),
    
    'r1_4term': lambda df: 3.301002567877740e+02 + \
                           -3.565682575738922e-01 * (df['ave:bulk_modulus'] + df['ave:gs_volume_per']) + \
                           3.786694583493326e-06 * (df['ave:vdw_radius_uff'] ** 3) + \
                           -1.705997650049823e+00 * (df['ave:gs_volume_per'] + df['ave:vdw_radius_uff']) + \
                           2.192371977941469e-01 * (df['ave:vdw_radius_uff'] * df['ave:gs_est_bcc_latcnt']),
    
    'r2_1term': lambda df: 9.732111258384252e+00 + \
                           1.163289662731223e-08 * ((df['ave:vdw_radius_uff']**3) * abs(df['ave:bulk_modulus'] - df['ave:atomic_radius_rahm'])),
    
    'r2_2term': lambda df: 1.173595476514455e+01 + \
                           4.740900052434680e+40 * (np.exp(-(df['ave:vdw_radius_uff']/df['ave:gs_est_bcc_latcnt']))) + \
                           1.133405246617076e-08 * ((df['ave:vdw_radius_uff']**3) * abs(df['ave:bulk_modulus'] - df['ave:atomic_radius_rahm'])),
    
    'r2_3term': lambda df: -4.694324548214459e+01 + \
                           -1.480067769463340e+00 * ((df['ave:vdw_radius_uff'] / df['ave:atomic_weight'])-(df['ave:first_ion_en'] / df['ave:en_ghosh'])) + \
                           4.829354094904532e+40 * (np.exp(-(df['ave:vdw_radius_uff']/df['ave:gs_est_bcc_latcnt']))) + \
                           1.123442733055649e-08 * ((df['ave:vdw_radius_uff']**3) * abs(df['ave:bulk_modulus'] - df['ave:atomic_radius_rahm'])),
    
    'r2_4term': lambda df: -4.818204304584309e+01 + \
                           4.425039768829772e+00 * (np.sin((df['ave:atomic_weight'] * df['ave:gs_est_bcc_latcnt']))) + \
                           -1.530621488950609e+00 * ((df['ave:vdw_radius_uff'] / df['ave:atomic_weight']) - (df['ave:first_ion_en'] / df['ave:en_ghosh'])) + \
                           4.774726643610366e+40 * (np.exp(-(df['ave:vdw_radius_uff']/df['ave:gs_est_bcc_latcnt']))) + \
                           1.115667532014910e-08 * ((df['ave:vdw_radius_uff']**3) * abs(df['ave:bulk_modulus'] - df['ave:atomic_radius_rahm']))
}

for key, fun in sisso_models.items():
    print(f"==========\nSISSO Model {key}")
    sisso_train_predictions = fun(sisso_data_train)
    sisso_test_predictions = fun(sisso_data_test)
    
    print("\nTest Set Error Metrics")
    for key, fun in metrics.items():
        value = fun(y_true=sisso_data_test['Volume'], y_pred=sisso_test_predictions)
        print(key,np.round(value,4))

    print("\nTraining Set Error Metrics")
    for key, fun in metrics.items():
        value = fun(y_true=sisso_data_train['Volume'], y_pred=sisso_train_predictions)
        print(key,np.round(value,4))
    
    
    sisso_data_train[key] = sisso_train_predictions
    sisso_data_test[key] = sisso_test_predictions


# It's not the best model, but we're gonna use the Rung1 1Term model, because it's simple and still performs well. It's also intuitive.

# In[]:


sisso_data_train.to_csv(os.path.join(sisso_dir, 'sisso_results_train.csv'))
sisso_data_test.to_csv(os.path.join(sisso_dir, 'sisso_results_test.csv'))


# In[]:


plt.scatter(x=sisso_data_train['Volume'], y=sisso_data_train['r1_1term'], label="Train Set")
plt.scatter(x=sisso_data_test['Volume'], y=sisso_data_test['r1_1term'], label="Test Set")

min_xy = min(min(sisso_data_train['Volume']),
             min(sisso_data_test['Volume']),
             min(sisso_data_train['r1_1term']),
             min(sisso_data_test['r1_1term']))
max_xy = max(max(sisso_data_train['Volume']),
             max(sisso_data_test['Volume']),
             max(sisso_data_train['r1_1term']),
             max(sisso_data_test['r1_1term']))

plt.plot([min_xy, max_xy], [min_xy, max_xy], label="Parity")
plt.ylabel(f"Perovskite Volume (Å^3 / formula unit) (Predicted)")
plt.xlabel(f"Perovskite Volume (Å^3 / formula unit) (Dataset)")
plt.legend()
plt.savefig("sisso_perovskite_volume_parity.jpeg")
plt.show()
plt.close()


# In[ ]:




