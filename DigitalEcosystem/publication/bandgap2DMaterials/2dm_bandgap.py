#!/usr/bin/env python
# coding: utf-8

# # 2D Material Bandgaps
# 
# In this notebook, we provide code to reproduce the results shown in our manuscript on the problem of predicting the badgap of 2D Materials using compositional and structural features.

# In[]:


import functools
import pickle
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
from DigitalEcosystem.utils.misc import root_mean_squared_error
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
# To start, we'll read in the data. Then, we'll filter out the dataset with the following rules:
# 
# 1. No elements from the f-block, anything larger than U, or noble gases
# 2. Decomposition energies must be below 0.5 eV
# 3. Exfoliation energies must be strictly positive
# 
# Also, because we utilize the Materials Project bandgap as a descriptor for the XGBoost and TPOT models, we necessarily must remove the bottom-up materials from the dataset. This is because only the top-down materials have corresponding entries on the Materials Project.

# In[]:


data = pd.read_pickle('../feature_engineering/full_featurized_data.pkl')


# Drop any missing entries
target_column = ["bandgap (eV)"]
data = data[data[target_column[0]].notna()]

# # Drop anything in the f-block, larger than U, and noble gases
bad_elements = noble_gases + f_block_elements + synthetic_elements_in_d_block
element_mask = data['atoms_object (unitless)'].apply(lambda atoms: all([forbidden not in atoms.get_chemical_symbols() for forbidden in bad_elements]))

# Drop anything that decomposes
decomposition_mask = data['decomposition_energy (eV/atom)'] < 0.5

# Drop things with non-strictly-positive exfoliation energies
exfol_mask = data['exfoliation_energy_per_atom (eV/atom)'] > 0

data = data[element_mask & decomposition_mask & exfol_mask]

# A parent Materials Project structure is not defined for the bottom-up materials. We waste our bandwith looking these up, so let's remove them.
data = data[data['discovery_process (unitless)'] != 'bottom-up']


# Next up we'll query the Materials Project through PyMatGen for the bandgap of each material

# In[]:


rester = pymatgen.ext.matproj.MPRester(api_key=os.getenv("MATERIALS_PROJECT_API_KEY"))


@except_with_default_value(default_return=None)
def get_mp_bandgap(structure):
    mp_id = DigitalEcosystem.utils.misc.get_parent_structure_id(structure)
    
    result = rester.get_data(mp_id, prop='band_gap')[0]['band_gap']
    return result
    
data['mp_bandgap'] = data['2dm_id (unitless)'].progress_apply(get_mp_bandgap)


# The above featurization takes a while, so let's just save it to disk. That way if we ever want to re-run this notebook, we don't have to re-do all those queries (and we save Materials Project some bandwidth)

# In[]:


data.to_pickle('filtered_data_with_bandgap.pkl')
#data = pd.read_pickle('filtered_data_with_bandgap.pkl')


# Any entries from Materials Project that were missing a bandgap can get thrown away

# In[]:


# Finally, we'll filter the data with missing values for bandgap
data = data[data['mp_bandgap'].notna()]


# In[]:


xenonpy_descriptors = [col for col in data.columns if ":" in col]

descriptors = xenonpy_descriptors + matminer_descriptors + ['mp_bandgap']


# Finally, we'll show the dataframe, to get an idea of how much data we've filtered out.

# In[]:


data[['2dm_id (unitless)'] + target_column + descriptors]


# # Prepare Data

# Next up, we'll perform a train/test split, holding out 10% of the data as a test set.

# In[]:


train, test = sklearn.model_selection.train_test_split(data, test_size=0.1, random_state=RANDOM_SEED)

train_x = np.nan_to_num(train[descriptors].to_numpy())
train_y = np.nan_to_num(train[target_column].to_numpy())

test_x = np.nan_to_num(test[descriptors].to_numpy())
test_y = np.nan_to_num(test[target_column].to_numpy())


# In[]:


metrics = {
    'MaxError': sklearn.metrics.max_error,
    'MAE': sklearn.metrics.mean_absolute_error,
    'MSE': sklearn.metrics.mean_squared_error,
    'RMSE': root_mean_squared_error,
    'MAPE': sklearn.metrics.mean_absolute_percentage_error,
    'R2': sklearn.metrics.r2_score
}


# # XGBoost
# 
# XGBoost is a gradient boosting algorithm that uses an ensemble of decision trees. It's a very flexible model that comes with a lot of hyperparameters to tune. To tune them, we'll use Optuna, a Bayesian optimization framework. We'll also use Optuna to choose whether we use Z-score normalization or min/max scaling on the data.
# 
# We'll hold out 20% of the data as a validation set, for early-stopping and pruning purposes. We'll train the model to minimize its RMSE on the training set.

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


DigitalEcosystem.utils.figures.save_parity_plot_publication_quality(train_y_true = train_y,
                                                                    train_y_pred = best_reg.predict(train_x),
                                                                    test_y_true = test_y,
                                                                    test_y_pred = best_reg.predict(test_x),
                                                                    axis_label = "Bandgap (eV)",
                                                                    filename = "xgboost_2dm_bandgap_parity.jpeg")


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

old_figsize = plt.rcParams["figure.figsize"]
plt.rcParams["figure.figsize"] = (2*old_figsize[0], old_figsize[1])

plt.barh(range(n_importances), [imp[0] for imp in sorted_importances[:n_importances]])
plt.yticks(range(n_importances), [imp[1] for imp in sorted_importances[:n_importances]])
plt.ylabel("Feature")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("xgboost_2dm_bandgap_importances.jpeg")
plt.rcParams["figure.figsize"] = old_figsize


# Finally, for some book-keeping purposes, we'll go ahead and save the predictions from the XGBoost model, along with the importance scores from the above plot. Also, we'll go ahead and pickle the XGBoost pipeline.

# In[]:


train_preds = train[target_column]
train_preds['TrainTest Status'] = ['Training Set'] * len(train_preds)
train_preds['Prediction'] = best_reg.predict(train_x)

test_preds = test[target_column]
test_preds['TrainTest Status'] = ['Test Set'] * len(test_preds)
test_preds['Prediction'] = best_reg.predict(test_x)

xgb_predictions = train_preds.append(test_preds)
xgb_predictions.to_csv("xgboost_2dm_bandgap_predictions.csv")


# In[]:


with open("xgboost_2dm_bandgap_importances.csv", "w") as outp:
    outp.write("Descriptor,XGB_Importance\n")
    for importance, descriptor in sorted_importances:
        outp.write(f"{descriptor},{importance}\n")


# In[]:


with open("xgboost_pipeline.pkl", "wb") as outp:
    pickle.dump(best_reg, outp)


# # TPOT
# 
# TPOT is an AutoML solution that uses a genetic algorithm to create an ML pipeline to address a given problem. Here, we'll run a population of 100 models over 10 generations, taking the 10-fold cross-validated RMSE as the fitness metric.
# 
# We'll also go ahead and save a parity plot of the TPOT model.

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


tpot_rr_coefs = zip(tpot_model.fitted_pipeline_[1].coef_, descriptors)
sorted_tpot_rr_coefs = list(sorted(tpot_rr_coefs, key=lambda i: -abs(i[0])))

old_figsize = plt.rcParams["figure.figsize"]
plt.rcParams["figure.figsize"] = (2*old_figsize[0], old_figsize[1])
print(sorted_tpot_rr_coefs)

plt.barh(range(n_importances), [imp[0] for imp in sorted_tpot_rr_coefs[:n_importances]])
plt.yticks(range(n_importances), [imp[1] for imp in sorted_tpot_rr_coefs[:n_importances]])
plt.ylabel("Feature")
plt.xlabel("Elastic Net Feature Coefficient")
plt.tight_layout()
plt.show()
plt.savefig("tpot_2dm_elasticnet_coefficients.jpeg")
plt.close()

plt.rcParams['figure.figsize'] = old_figsize


# In[]:


DigitalEcosystem.utils.figures.save_parity_plot_publication_quality(train_y_true = train_y,
                                                                    train_y_pred = tpot_model.predict(train_x),
                                                                    test_y_true = test_y,
                                                                    test_y_pred = tpot_model.predict(test_x),
                                                                    axis_label = "Bandgap (eV))",
                                                                    filename = "tpot_2dm_bandgap_parity.jpeg")


# In[]:


print("Test Set Error Metrics")
for key, fun in metrics.items():
    value = fun(y_true=test_y, y_pred=tpot_model.predict(test_x))
    print(key,np.round(value,4))
    
print("\nTraining Set Error Metrics")
for key, fun in metrics.items():
    value = fun(y_true=train_y, y_pred=tpot_model.predict(train_x))
    print(key,np.round(value,4))


# Finally, we'll go ahead and back up those predictions to the disk (this way, we don't need to re-run this again just to get those), and we'll pickle the TPOT model. We'll also have TPOT auto-generate some Python code to re-train itself.

# In[]:


train_preds = train[target_column]
train_preds['TrainTest Status'] = ['Training Set'] * len(train_preds)
train_preds['Prediction'] = tpot_model.predict(train_x)

test_preds = test[target_column]
test_preds['TrainTest Status'] = ['Test Set'] * len(test_preds)
test_preds['Prediction'] = tpot_model.predict(test_x)

tpot_predictions = train_preds.append(test_preds)
tpot_predictions.to_csv("tpot_2dm_bandgap_predictions.csv")


# At this point, a SISSO model was run. The models are stored below.
# 
# The model forms are from the SISSO logfiles. Coefficients have been copied directly from the SISSO outputs here.

# In[]:


tpot_model.export('tpot_autogenerated_pipeline.py')
with open("tpot_pipeline.pkl", "wb") as outp:
    pickle.dump(tpot_model.fitted_pipeline_, outp)


# # Roost
# 
# [Roost](https://github.com/CompRhys/roost) is a neural network approach to predicting material properties as a function of their composition. Although we only have 144 data-points here, we can at least try for a good model.
# 
# Since the model only requires material IDs, the composition, and the property of interest, we'll save a CSV containing those properties.

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


DigitalEcosystem.utils.figures.save_parity_plot_publication_quality(train_y_true = roost_train_results['bandgap_target'],
                                                                    train_y_pred =  roost_train_results['bandgap_pred_n0'],
                                                                    test_y_true = roost_test_results['bandgap_target'],
                                                                    test_y_pred = roost_test_results['bandgap_pred_n0'],
                                                                    axis_label = "Bandgap (eV)",
                                                                    filename = "roost_2dm_bandgap_parity.jpeg")


# In[]:


print("Test Set Error Metrics")
for key, fun in metrics.items():
    value = fun(y_true=roost_test_results['bandgap_target'], y_pred=roost_test_results['bandgap_pred_n0'])
    print(key,np.round(value,4))
    
print("\nTraining Set Error Metrics")
for key, fun in metrics.items():
    value = fun(y_true=roost_train_results['bandgap_target'], y_pred=roost_train_results['bandgap_pred_n0'])
    print(key,np.round(value,4))


# # SISSO
# 
# SISSO is a symbolic regression technique focused on creating interpretable machine learning models. 
# 
# Due to the exponential computational cost of running a SISSO model as the number of features and rungs increases, we need to restrict the feature space. To do that, we'll use LASSO-based feature selection (essentially we can look at how quickly LASSO extinguishes a variable to get an idea of its importance). 

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
sisso_data_test.to_csv(os.path.join(sisso_dir, 'sisso_test.csv'), index_label='2dm_id (unitless)')


# At this point, a SISSO model was run. The models are stored below.
# 
# The model forms are from the SISSO logfiles. For example, the "r1_1term" model corresponds with the 1-term model from rung 1.
# 
# The coefficients are extracted from the generated model `.dat` files, found in the `sisso/models` directory. 

# In[]:


sisso_models = {
    'r1_1term': lambda df: 5.865590038114292e+00 + \
                           -4.133268583136611e-01 * (np.cbrt(df['ave:boiling_point'])),
    
    'r1_2term': lambda df: 5.456824304436886e+00 + \
                           5.806272825102222e-05 * (df['var:sound_velocity'] / df['ave:boiling_point']) + \
                           -3.928478305199087e-01 * (np.cbrt(df['ave:boiling_point'])),
    
    'r1_3term': lambda df: 5.829318713354136e+00 + \
                           -2.895404136895635e-02 * (df['var:thermal_conductivity'] / df['ave:boiling_point']) + \
                           5.661787713744031e-05 * (df['var:sound_velocity'] / df['ave:boiling_point']) + \
                           -4.133601815903408e-01 * (np.cbrt(df['ave:boiling_point'])),
    
    'r1_4term': lambda df: 4.239649765792811e+00 + \
                           3.209844715367811e-04 * (df['sum:hhi_p'] / df['ave:atomic_weight']) + \
                           -2.931356075881642e-02 * (df['var:thermal_conductivity'] / df['ave:boiling_point']) + \
                           5.565612620549401e-06 * (df['var:sound_velocity'] / df['ave:atomic_radius_rahm']) + \
                           -8.322616418822797e-02 * (np.sqrt(df['ave:boiling_point'])),
    
    'r2_1term': lambda df: -6.848927418726274e-01 + \
                           1.020330482802878e+00 * ((abs(df['ave:atomic_volume'] - df['ave:Polarizability'])) / (np.sqrt(df['ave:atomic_number']))),
    
    'r2_2term': lambda df: -7.037029355457204e-01 + \
                           -1.319062598081636e-03 * ((df['var:thermal_conductivity'] * df['ave:bulk_modulus']) / (df['ave:boiling_point'] * df['ave:Polarizability'])) + \
                           1.084886105825630e+00 * ((abs(df['ave:atomic_volume'] - df['ave:Polarizability'])) / (np.sqrt(df['ave:atomic_number']))),
    
    'r2_3term': lambda df: 1.288364460133121e+00 + \
                           -4.685654009911847e-02 * ((df['ave:atomic_number'] / df['ave:atomic_weight']) * (df['ave:atomic_radius_rahm'] - df['ave:atomic_radius'])) + \
                           -3.019449625791471e-05 * ((df['var:thermal_conductivity'] * df['var:boiling_point']) / (df['ave:boiling_point']**2)) + \
                           1.284011701624066e+00 * ((abs(df['ave:atomic_volume'] - df['ave:Polarizability'])) / (np.sqrt(df['ave:atomic_weight']))),
    
    'r2_4term': lambda df: 1.264193645822197e+00 + \
                           1.670730994972521e+01 * ((np.cbrt(df['var:thermal_conductivity'])) / (df['ave:bulk_modulus'] * df['ave:atomic_volume'])) + \
                           -5.358316290237519e-02 * ((df['ave:atomic_number'] / df['ave:atomic_weight']) * (df['ave:atomic_radius_rahm'] - df['ave:atomic_radius'])) + \
                           -3.794450825704627e-05 * ((df['var:thermal_conductivity'] * df['var:boiling_point']) / (df['ave:boiling_point']**2)) + \
                           1.281828774755674e+00 * ((abs(df['ave:atomic_volume'] - df['ave:Polarizability'])) / (np.sqrt(df['ave:atomic_weight'])))
}

for key, fun in sisso_models.items():
    print(f"==========\nSISSO Model {key}")
    sisso_train_predictions = fun(sisso_data_train)
    sisso_test_predictions = fun(sisso_data_test)
    sisso_data_train[key] = sisso_train_predictions
    sisso_data_test[key] = sisso_test_predictions
    
    print("\nTest Set Error Metrics")
    for metric, fun in metrics.items():
        value = fun(y_true=sisso_data_test['bandgap (eV)'], y_pred=sisso_test_predictions)
        print(metric,np.round(value,4))

    print("\nTraining Set Error Metrics")
    for metric, fun in metrics.items():
        value = fun(y_true=sisso_data_train['bandgap (eV)'], y_pred=sisso_train_predictions)
        print(metric,np.round(value,4))
    
    


# Finally, we'll go ahead and save the predictions of the SISSO model on the training and test set.

# In[]:


sisso_data_train.to_csv(os.path.join(sisso_dir, 'sisso_results_train.csv'))
sisso_data_test.to_csv(os.path.join(sisso_dir, 'sisso_results_test.csv'))


# In[]:


model_to_plot = 'r2_4term'
DigitalEcosystem.utils.figures.save_parity_plot_publication_quality(train_y_true = sisso_data_train['bandgap (eV)'],
                                                                    train_y_pred = sisso_data_train[model_to_plot],
                                                                    test_y_true = sisso_data_test['bandgap (eV)'],
                                                                    test_y_pred = sisso_data_test[model_to_plot],
                                                                    axis_label = "Bandgap (eV)",
                                                                    filename = "sisso_2dm_bandgap_parity.jpeg")


# Finally, just so we have them, let's print out the rest of the SISSO models

# In[]:


for model_to_plot in sisso_models.keys():
    DigitalEcosystem.utils.figures.save_parity_plot_publication_quality(train_y_true = sisso_data_train['bandgap (eV)'],
                                                                        train_y_pred = sisso_data_train[model_to_plot],
                                                                        test_y_true = sisso_data_test['bandgap (eV)'],
                                                                        test_y_pred = sisso_data_test[model_to_plot],
                                                                        axis_label = "Bandgap (eV)",
                                                                        title=f'SISSO Rung-{model_to_plot[1]}, {model_to_plot[3]}-term Model')


# In[]:


import scipy

for i in [imp[1] for imp in sorted_importances[:5]]:
    print(i)
    print(scipy.stats.pearsonr(x=np.nan_to_num(data[i].to_numpy()), y=data['bandgap (eV)']))
    print(scipy.stats.spearmanr(a=np.nan_to_num(data[i].to_numpy()), b=data['bandgap (eV)']))
    print('========')


# In[ ]:




