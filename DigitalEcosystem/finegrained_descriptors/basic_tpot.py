#!/usr/bin/env python
# coding: utf-8

# In[]:


import pandas as pd
import sklearn.model_selection
import matplotlib.pyplot as plt

import tpot


# In[]:


RANDOM_SEED=1234


# In[]:


datapath = "../DigitalEcosystem/all_data_features.pkl"
all_data = pd.read_pickle(datapath)


# In[]:


all_data.drop(columns=["atoms_object (unitless)", "pymatgen_structure (unitless)"])


# In[]:


data_train, data_test = sklearn.model_selection.train_test_split(all_data.drop(columns=["atoms_object (unitless)", "pymatgen_structure (unitless)"]),
                                                                 random_state=RANDOM_SEED,
                                                                 test_size=0.2)

target = "bandgap (eV)"
train_x = data_train.drop(columns=[target]).to_numpy()
train_y = data_train[target].to_numpy()

test_x = data_test.drop(columns=[target]).to_numpy()
test_y = data_test[target].to_numpy()


# In[]:


model = tpot.TPOTRegressor(
    generations=None,
    population_size=100,
    max_eval_time_mins=10/60,
    max_time_mins=20,
    cv=5,
    verbosity=2,
    scoring="r2",
    config_dict=tpot.config.regressor_config_dict,
    n_jobs=4,
    random_state=RANDOM_SEED
)
model.fit(features=train_x, target=train_y)


# In[]:


train_y_pred = model.predict(train_x)
test_y_pred = model.predict(test_x)
plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
plt.rcParams["figure.figsize"] = (10,10)

plt.scatter(x=train_y_pred, y=train_y, label="Train Set")
plt.scatter(x=test_y_pred, y=test_y, label="Test Set")
plt.plot([0,8], [0,8], label="Parity")
plt.ylabel("Bandgap (Actual)")
plt.xlabel("Bandgap (Predicted)")
plt.legend()


# In[]:


import ase


# In[]:


ase.__version__


# In[ ]:




