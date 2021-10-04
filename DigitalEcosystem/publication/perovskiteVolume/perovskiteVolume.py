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
                                                "xgboost_perovskite_volume_parity.jpeg")


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

roost_data_train.to_csv(os.path.join(roost_dir, 'roost_train.csv'))
roost_data_test.to_csv(os.path.join(roost_dir, 'roost_test.csv'))


# # SISSO

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # SISSO Models
# Here, we have Rung1 and Rung2 models that have been generated by SISSO using the XenonPy compositional descriptors. 

# In[ ]:


models = {
    "r1_1term": lambda df: 2.342082515585748e-02 + \
                           5.172456498122173e-01 * (df["ave:vdw_radius_uff"] + df["ave:covalent_radius_pyykko_double"]),
    "r1_2term": lambda df: 1.052443572291616e-02 + \
                           1.242091145866421e-01 * (df["ave:num_p_valence"] + df["ave:num_p_unfilled"]) + \
                           4.898720001428966e-01 * (df["min:gs_volume_per"] + df["ave:covalent_radius_pyykko_double"]),
    "r2_1term": lambda df: 4.521699008847579e-03 + \
                           2.966651096729857e-01 * (
                                   (df["min:Polarizability"] + df["ave:num_p_valence"]) + \
                                   (df["min:gs_volume_per"] + df["ave:vdw_radius_mm3"])
                           )
}

data_train_scaled_sisso = data_train_scaled.copy()
data_test_scaled_sisso = data_test_scaled.copy()
for key, fun in models.items():
    data_train_scaled_sisso[key] = fun(data_train_scaled_sisso)
    data_test_scaled_sisso[key] = fun(data_test_scaled_sisso)

data_train_scaled_sisso.head()


# # TPOT Model
# 
# To have something to compare to, we'll also run a TPOT model for 10 minutes.

# In[ ]:


# TPOT with the vanilla features

model = tpot.TPOTRegressor(
    generations=None,
    population_size=100,
    max_eval_time_mins=1 / 60,
    max_time_mins=10,
    cv=len(data_train_scaled),
    verbosity=2,
    scoring="neg_root_mean_squared_error",
    config_dict=tpot.config.regressor_config_dict,
    n_jobs=-1,
    random_state=1234
)


# In[ ]:


train_x = data_train_scaled.drop(columns="Volume").to_numpy()
train_y = data_train_scaled.Volume.to_numpy().ravel()

test_x = data_test_scaled.drop(columns="Volume").to_numpy()
test_y = data_test_scaled.Volume.to_numpy().ravel()


# ## TPOT Pipeline
# TPOT Generates the following pipeline:
# 1. Decision Tree Regressor
# 2. The decision tree's predictions are scaled to be between 0 and 1.
# 3. Ridge regression is used on this scaled set of predictions.
# 
# ## TPOT Model Plot

# In[ ]:


def unscale(arr):
    return arr * std["Volume"] + mean["Volume"]


train_pred_y = model.predict(train_x)
test_pred_y = model.predict(test_x)

create_parity_plot_with_raw_values = functools.partial(DigitalEcosystem.utils.figures.save_parity_plot_with_raw_values,
                                                       train_y=unscale(train_y),
                                                       test_y=unscale(test_y))

create_parity_plot_with_raw_values(unscale(train_pred_y),
                                   unscale(test_pred_y),
                                   filename='tpot_parity.png')


# # SISSO Rung1, 1-Term Plot

# In[ ]:


create_parity_plot_with_raw_values(unscale(data_train_scaled_sisso["r1_1term"]),
                                   unscale(data_test_scaled_sisso["r1_1term"]),
                                   filename='sisso_r1_1t_parity.png')


# # SISSO Rung1, 2-Term Plot

# In[ ]:


create_parity_plot_with_raw_values(unscale(data_train_scaled_sisso["r1_2term"]),
                                   unscale(data_test_scaled_sisso["r1_2term"]),
                                   filename='sisso_r1_2t_parity.png')


# # SISSO Rung2, 1-Term Plot

# In[ ]:


create_parity_plot_with_raw_values(unscale(data_train_scaled_sisso["r2_1term"]),
                                   unscale(data_test_scaled_sisso["r2_1term"]),
                                   filename='sisso_r2_1t_parity.png')


# # Combined Plots
# 
# Finally, we compare the results of the SISSO models and the TPOT model.
# 
# ## Training Set
# 
# Below, we plot the Training set results for:
# 1. The TPOT model (Black + symbols)
# 2. The SISSO Rung1, 1-Term Model (Red Nablas / Upside-Down Triangles)
# 3. The SISSO Rung1, 2-Term Model (Green Deltas / Rightside-Up Triangles)
# 4. The SISSO Rung2, 1-Term Model (Blue Squares)
# 
# As a guide to the eye, parity is also drawn as a dashed black line.

# In[ ]:


tpot_mape = np.round(sklearn.metrics.mean_absolute_percentage_error(y_true=unscale(train_y),  y_pred=unscale(train_pred_y)),2)
r1_1t_mape = np.round(sklearn.metrics.mean_absolute_percentage_error(y_true=unscale(train_y), y_pred=unscale(data_train_scaled_sisso["r1_1term"])), 2)
r1_2t_mape = np.round(sklearn.metrics.mean_absolute_percentage_error(y_true=unscale(train_y), y_pred=unscale(data_train_scaled_sisso["r1_2term"])), 2)
r2_1t_mape = np.round(sklearn.metrics.mean_absolute_percentage_error(y_true=unscale(train_y), y_pred=unscale(data_train_scaled_sisso["r2_1term"])), 2)

alphas = (0.9, 0.5, 0.5, 0.5)
markers = '+v^s'
colors=('black','red', 'green', 'blue')
labels=(f"TPOT, 108 Terms, MAPE={tpot_mape}",
        f"Rung 1, 1-Term, MAPE={r1_1t_mape}",
        f"Rung 1, 2-Term, MAPE={r1_2t_mape}",
        f"Rung 2, 1-term, MAPE={r2_1t_mape}")

DigitalEcosystem.utils.figures.create_multi_parity_plot(ytrue=unscale(train_y), 
                                                        series_to_plot = map(unscale, (train_pred_y, 
                                                                                       data_train_scaled_sisso['r1_1term'],
                                                                                       data_train_scaled_sisso['r1_2term'],
                                                                                       data_train_scaled_sisso['r2_1term'])
                                                                            ),
                                                        markers=markers,
                                                        alphas=alphas,
                                                        colors=colors,
                                                        labels=labels,
                                                        is_train=True)


# ## Test-Set
# Below, we plot the Training set results for:
# 1. The TPOT model (Black + symbols)
# 2. The SISSO Rung1, 1-Term Model (Red Nablas / Upside-Down Triangles)
# 3. The SISSO Rung1, 2-Term Model (Green Deltas / Rightside-Up Triangles)
# 4. The SISSO Rung2, 1-Term Model (Blue Squares)
# 
# As a guide to the eye, parity is also drawn as a dashed black line.

# In[ ]:


DigitalEcosystem.utils.figures.create_multi_parity_plot(ytrue=unscale(test_y), 
                                                        series_to_plot = map(unscale, (test_pred_y, 
                                                                                       data_test_scaled_sisso['r1_1term'],
                                                                                       data_test_scaled_sisso['r1_2term'],
                                                                                       data_test_scaled_sisso['r2_1term'])
                                                                            ),
                                                        markers=markers,
                                                        alphas=alphas,
                                                        colors=colors,
                                                        labels=labels,
                                                        is_train=False)


# # Conclusions
# 
# Overall, we find that SISSO creates models that are much simpler than the TPOT pipeline, while being nearly as performant. Generally, they seem to rely heavily on atomic volume descriptors.
# 
# Although this may initially seem like an obvious result (as naturally, the crystal cell should correlated with the atomic volumes contained within), keep in mind that SISSO took in no other information about this result. Moreover, it generates easy-to-interpret linear models that give the user "knobs" that can be turned to effect some result.
# 
# Contrast with the TPOT model, which contains a decision tree followed by some postprocessing followed by ridge regression. This model is essentially a black box - although the decision trees are generally interpretable, we're also heavily postprocessing it. Morevoer, a decision tree is an interpolative model, which means we'd be unable to extrapolate. The decision tree also takes in over 100 features.
# 
# Contrary to this limitation, the SISSO model will extrapolate, takes in just 2-4 features (depending on the rung and number of terms), and results to something that's very close to the conventional wisdom a chemist might offer.
# 
# Overall, although the TPOT model performs *slightly* better, the SISSO model has more utility if one seeks to obtain physical insight.

# In[ ]:




