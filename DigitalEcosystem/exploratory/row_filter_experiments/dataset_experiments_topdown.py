#!/usr/bin/env python
# coding: utf-8

# In[]:


import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.model_selection

import tpot


# In[]:


MAX_TIME = 5 # minutes
RANDOM_SEED = 1234


# In[]:


csv_files = [i for i in os.listdir() if i.endswith("csv")]
dataframes = [pd.read_csv(file, index_col="Unnamed: 0") for file in csv_files]

experiments = {}
for filename, dataframe in zip(csv_files, dataframes):
    
    dataframe = dataframe[dataframe["discovery_process (unitless)"] == "top-down"]
    
    if len(dataframe) > 0:
        experiments[filename] = dataframe
print("\n".join(experiments.keys()))


# In[]:


targets = ["bandgap (eV)", "exfoliation_energy_per_atom (eV/atom)"]
n_models = len(experiments.keys()) * len(targets)

print(f"Estimated time to evaluate all {n_models} scenarios, budgeting {MAX_TIME} minutes per model: {n_models * MAX_TIME} minutes")


# In[]:


def experiment_generator():
    for name, df in experiments.items():
        for target in targets:
            yield name, df, target
experiment = experiment_generator()

def run_experiment(name, df, target):
    test_size = 0.1
    regression_irrelevant = [
        '2dm_id (unitless)',
        'formula',
        'discovery_process (unitless)',
        'potcars (unitless)',
        'is_hubbard (unitless)',
        'energy_per_atom (eV)',
        'decomposition_energy (eV/atom)',
        'is_bandgap_direct (unitless)',
        'is_metal (unitless)',
        'energy_vdw_per_atom (eV/atom)',
        'total_magnetization (Bohr Magneton)']
    
    # Ignore other target columns
    other_targets = [col for col in targets if col != target]
    df = df.drop(columns=other_targets + regression_irrelevant)
    
    # Train / Test Split
    data_train, data_test = sklearn.model_selection.train_test_split(df, test_size=test_size, random_state=RANDOM_SEED)
    train_x = data_train.drop(columns=target).to_numpy()
    train_y = data_train[target].to_numpy()
    
    test_x = data_test.drop(columns=target).to_numpy()
    test_y = data_test[target].to_numpy()
    
    # Train a model
    model = tpot.TPOTRegressor(
        generations=None,
        population_size=100,
        max_eval_time_mins=5/60,
        max_time_mins=MAX_TIME,
        cv=4,
        verbosity=2,
        scoring="r2",
        config_dict=tpot.config.regressor_config_dict,
        n_jobs=22,
        random_state=RANDOM_SEED
    )
    model.fit(features=train_x, target=train_y)
    
    train_y_pred = model.predict(train_x)
    test_y_pred = model.predict(test_x)
    
    print("Test-Set Error Metrics:")
    for name, metric in [
        ["MAE", sklearn.metrics.mean_absolute_error],
        ["MAPE", sklearn.metrics.mean_absolute_percentage_error],
        ["MSE", sklearn.metrics.mean_squared_error],
        ["R2", sklearn.metrics.r2_score],
        ["Max Error", sklearn.metrics.max_error]
    ]:
        print(f"{name}: {np.round(metric(y_true=test_y, y_pred=test_y_pred), 3)}")
    
    # Plot the results   
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
    plt.rcParams["figure.figsize"] = (10,10)
    plt.rcParams["font.size"] = 16

    plt.scatter(x=train_y_pred, y=train_y, label="Train Set")
    plt.scatter(x=test_y_pred, y=test_y, label="Test Set")
    
    min_xy = min(min(train_y_pred), min(train_y))
    max_xy = max(max(train_y_pred), max(train_y))
    
    plt.plot([min_xy,max_xy], [min_xy,max_xy], label="Parity")
    plt.ylabel(f"{target} (Dataset)")
    plt.xlabel(f"{target} (Predicted)")
    plt.legend()
    plt.show()


# In[]:


name, df, target = next(experiment)
print(f"Name: {name}")
print(f"Rows: {len(df)}")
print(f"Target: {target}")
try:
    run_experiment(name, df, target)
except Exception as e:
    print("Something went wrong, continuing down to the other experiments. Traceback:")
    print(e)


# In[]:


name, df, target = next(experiment)
print(f"Name: {name}")
print(f"Rows: {len(df)}")
print(f"Target: {target}")
try:
    run_experiment(name, df, target)
except Exception as e:
    print("Something went wrong, continuing down to the other experiments. Traceback:")
    print(e)


# In[]:


name, df, target = next(experiment)
print(f"Name: {name}")
print(f"Rows: {len(df)}")
print(f"Target: {target}")
try:
    run_experiment(name, df, target)
except Exception as e:
    print("Something went wrong, continuing down to the other experiments. Traceback:")
    print(e)


# In[]:


name, df, target = next(experiment)
print(f"Name: {name}")
print(f"Rows: {len(df)}")
print(f"Target: {target}")
try:
    run_experiment(name, df, target)
except Exception as e:
    print("Something went wrong, continuing down to the other experiments. Traceback:")
    print(e)


# In[]:


name, df, target = next(experiment)
print(f"Name: {name}")
print(f"Rows: {len(df)}")
print(f"Target: {target}")
try:
    run_experiment(name, df, target)
except Exception as e:
    print("Something went wrong, continuing down to the other experiments. Traceback:")
    print(e)


# In[]:


name, df, target = next(experiment)
print(f"Name: {name}")
print(f"Rows: {len(df)}")
print(f"Target: {target}")
try:
    run_experiment(name, df, target)
except Exception as e:
    print("Something went wrong, continuing down to the other experiments. Traceback:")
    print(e)


# In[]:


name, df, target = next(experiment)
print(f"Name: {name}")
print(f"Rows: {len(df)}")
print(f"Target: {target}")
try:
    run_experiment(name, df, target)
except Exception as e:
    print("Something went wrong, continuing down to the other experiments. Traceback:")
    print(e)


# In[]:


name, df, target = next(experiment)
print(f"Name: {name}")
print(f"Rows: {len(df)}")
print(f"Target: {target}")
try:
    run_experiment(name, df, target)
except Exception as e:
    print("Something went wrong, continuing down to the other experiments. Traceback:")
    print(e)


# In[]:


name, df, target = next(experiment)
print(f"Name: {name}")
print(f"Rows: {len(df)}")
print(f"Target: {target}")
try:
    run_experiment(name, df, target)
except Exception as e:
    print("Something went wrong, continuing down to the other experiments. Traceback:")
    print(e)


# In[]:


name, df, target = next(experiment)
print(f"Name: {name}")
print(f"Rows: {len(df)}")
print(f"Target: {target}")
try:
    run_experiment(name, df, target)
except Exception as e:
    print("Something went wrong, continuing down to the other experiments. Traceback:")
    print(e)


# In[]:


name, df, target = next(experiment)
print(f"Name: {name}")
print(f"Rows: {len(df)}")
print(f"Target: {target}")
try:
    run_experiment(name, df, target)
except Exception as e:
    print("Something went wrong, continuing down to the other experiments. Traceback:")
    print(e)


# In[]:


name, df, target = next(experiment)
print(f"Name: {name}")
print(f"Rows: {len(df)}")
print(f"Target: {target}")
try:
    run_experiment(name, df, target)
except Exception as e:
    print("Something went wrong, continuing down to the other experiments. Traceback:")
    print(e)


# In[]:


name, df, target = next(experiment)
print(f"Name: {name}")
print(f"Rows: {len(df)}")
print(f"Target: {target}")
try:
    run_experiment(name, df, target)
except Exception as e:
    print("Something went wrong, continuing down to the other experiments. Traceback:")
    print(e)


# In[]:


name, df, target = next(experiment)
print(f"Name: {name}")
print(f"Rows: {len(df)}")
print(f"Target: {target}")
try:
    run_experiment(name, df, target)
except Exception as e:
    print("Something went wrong, continuing down to the other experiments. Traceback:")
    print(e)


# In[]:


name, df, target = next(experiment)
print(f"Name: {name}")
print(f"Rows: {len(df)}")
print(f"Target: {target}")
try:
    run_experiment(name, df, target)
except Exception as e:
    print("Something went wrong, continuing down to the other experiments. Traceback:")
    print(e)


# In[]:


name, df, target = next(experiment)
print(f"Name: {name}")
print(f"Rows: {len(df)}")
print(f"Target: {target}")
try:
    run_experiment(name, df, target)
except Exception as e:
    print("Something went wrong, continuing down to the other experiments. Traceback:")
    print(e)


# In[]:


name, df, target = next(experiment)
print(f"Name: {name}")
print(f"Rows: {len(df)}")
print(f"Target: {target}")
try:
    run_experiment(name, df, target)
except Exception as e:
    print("Something went wrong, continuing down to the other experiments. Traceback:")
    print(e)


# In[]:


name, df, target = next(experiment)
print(f"Name: {name}")
print(f"Rows: {len(df)}")
print(f"Target: {target}")
try:
    run_experiment(name, df, target)
except Exception as e:
    print("Something went wrong, continuing down to the other experiments. Traceback:")
    print(e)


# In[]:


name, df, target = next(experiment)
print(f"Name: {name}")
print(f"Rows: {len(df)}")
print(f"Target: {target}")
try:
    run_experiment(name, df, target)
except Exception as e:
    print("Something went wrong, continuing down to the other experiments. Traceback:")
    print(e)


# In[]:


name, df, target = next(experiment)
print(f"Name: {name}")
print(f"Rows: {len(df)}")
print(f"Target: {target}")
try:
    run_experiment(name, df, target)
except Exception as e:
    print("Something went wrong, continuing down to the other experiments. Traceback:")
    print(e)


# In[]:


name, df, target = next(experiment)
print(f"Name: {name}")
print(f"Rows: {len(df)}")
print(f"Target: {target}")
try:
    run_experiment(name, df, target)
except Exception as e:
    print("Something went wrong, continuing down to the other experiments. Traceback:")
    print(e)


# In[ ]:


name, df, target = next(experiment)
print(f"Name: {name}")
print(f"Rows: {len(df)}")
print(f"Target: {target}")
try:
    run_experiment(name, df, target)
except Exception as e:
    print("Something went wrong, continuing down to the other experiments. Traceback:")
    print(e)


# In[ ]:


name, df, target = next(experiment)
print(f"Name: {name}")
print(f"Rows: {len(df)}")
print(f"Target: {target}")
try:
    run_experiment(name, df, target)
except Exception as e:
    print("Something went wrong, continuing down to the other experiments. Traceback:")
    print(e)


# In[ ]:


name, df, target = next(experiment)
print(f"Name: {name}")
print(f"Rows: {len(df)}")
print(f"Target: {target}")
try:
    run_experiment(name, df, target)
except Exception as e:
    print("Something went wrong, continuing down to the other experiments. Traceback:")
    print(e)


# In[ ]:


name, df, target = next(experiment)
print(f"Name: {name}")
print(f"Rows: {len(df)}")
print(f"Target: {target}")
try:
    run_experiment(name, df, target)
except Exception as e:
    print("Something went wrong, continuing down to the other experiments. Traceback:")
    print(e)


# In[ ]:




