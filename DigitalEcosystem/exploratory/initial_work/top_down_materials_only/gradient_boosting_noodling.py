#!/usr/bin/env python
# coding: utf-8

# In[]:


import pandas as pd


# In[]:


import math
import random
import numpy as np
import sklearn
import sklearn.ensemble
import pandas as pd
import matplotlib.pyplot as plt
import tpot


# In[]:


data = pd.read_pickle("../raw_data/2d_mat_dataset_raw.pkl")
data = data.drop(columns=["2dm_id (unitless)", "formula", "atoms_object (unitless)", "potcars (unitless)", "is_hubbard (unitless)", "is_bandgap_direct (unitless)", "energy_per_atom (eV)", "is_metal (unitless)",
                          "decomposition_energy (eV/atom)"]).dropna().drop(columns=["discovery_process (unitless)"])
#data = data[data["discovery_process (unitless)"] == "top-down"].drop(columns=["discovery_process (unitless)"])
data


# In[]:


random.seed(1234)
np.random.seed(1234)
data_train, data_test = sklearn.model_selection.train_test_split(data, test_size=0.2)
mean, std = data_train.mean(), data_train.std()

data_train_scaled = (data_train - mean) / std
data_test_scaled = (data_test - mean) / std


# In[]:


model = tpot.TPOTRegressor(
    generations=None,
    population_size=100,
    max_eval_time_mins=1/60,
    max_time_mins=10,
    cv=10,
    verbosity=2,
    scoring="neg_root_mean_squared_error",
    config_dict=tpot.config.regressor_config_dict,
    n_jobs=-1,
    random_state=1234
)


# In[]:


#newmod = model
newmod


# In[]:


model = sklearn.ensemble.GradientBoostingRegressor(n_estimators=200, max_depth=5)


# In[]:


target = "bandgap (eV)"
train_x = np.nan_to_num(data_train_scaled.drop(columns=[target]).to_numpy())
train_y = np.nan_to_num(data_train_scaled[target].to_numpy().ravel())

test_x = np.nan_to_num(data_test_scaled.drop(columns=[target]).to_numpy())
test_y = np.nan_to_num(data_test_scaled[target].to_numpy().ravel())

model.fit(train_x, train_y)


# In[]:


def unscale(arr):
    return arr * std[target] + mean[target]
    
train_pred_y = model.predict(train_x)
test_pred_y = model.predict(test_x[:340,:])


# In[]:


models = {}

models["r1_1term"] = lambda df: -7.517588190691672e-03 + 3.019120658205411e-01 * (df["ave:fusion_enthalpy"] - df["ave:electron_affinity"])
models["r1_2term"] = lambda df: -4.220354429619663e-01 + 3.043650335588696e-01 * np.exp(df["sum:gs_energy"]) +                                  3.290126411913264e-01 * (df["ave:fusion_enthalpy"] - df["ave:electron_affinity"])
models["r2_1term"] = lambda df: -4.730850253495772e-01 + 3.491171896552910e-01 * (np.exp(df["sum:gs_energy"]) + (df["ave:fusion_enthalpy"] - df["ave:electron_affinity"]))

data_train_scaled_sisso = data_train_scaled.copy()
data_test_scaled_sisso = data_test_scaled.copy()
for key,fun in models.items():
    data_train_scaled_sisso[key] = fun(data_train_scaled_sisso)
    data_test_scaled_sisso[key] = fun(data_test_scaled_sisso)

data_train_scaled_sisso.head()


# In[]:


tpot_mae =  np.round(sklearn.metrics.mean_absolute_error(y_true=unscale(train_y), y_pred=unscale(train_pred_y)),2)
r1_1t_mae = np.round(sklearn.metrics.mean_absolute_error(y_true=unscale(train_y), y_pred=unscale(data_train_scaled_sisso["r1_1term"])),2)
r1_2t_mae = np.round(sklearn.metrics.mean_absolute_error(y_true=unscale(train_y), y_pred=unscale(data_train_scaled_sisso["r1_2term"])),2)
r2_1t_mae = np.round(sklearn.metrics.mean_absolute_error(y_true=unscale(train_y), y_pred=unscale(data_train_scaled_sisso["r2_1term"])),2)

plt.rcParams["figure.dpi"]=200
fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, sharey=True)
ax1.set_ylabel("Actual DFT Decomposition Energy (eV)")
ax2.set_xlabel("Predicted DFT Decomposition Energy (eV)")


    
ax1.scatter(x=unscale(data_train_scaled_sisso["r1_1term"]), y=unscale(train_y), marker="v", color="red",alpha=0.2, label=f"Rung 1, 1-Term, MAE={r1_1t_mae}")
ax2.scatter(x=unscale(data_train_scaled_sisso["r1_2term"]), y=unscale(train_y), marker="^", color="green", alpha=0.2, label=f"Rung 1, 2-Term, MAE={r1_2t_mae}")
ax3.scatter(x=unscale(data_train_scaled_sisso["r2_1term"]), y=unscale(train_y), marker="s", color="blue", alpha=0.2, label=f"Rung 2, 1-term, MAE={r2_1t_mae}")
for ax in (ax1, ax2, ax3):
    ax.scatter(x=unscale(train_pred_y), y=unscale(train_y), color="black", alpha=0.3, marker="+", label=f"TPOT, 108 Terms, MAE={tpot_mae}")
    ax.plot([0, 3], [0, 3], color="black", linestyle="--", label="Parity")
    
for ax in (ax1, ax2, ax3):
    ax.legend(prop={"size":5}, loc="upper center")

ax2.set_title("Training Set (80% of Dataset)")
plt.show()


# In[]:


tpot_mae =  np.round(sklearn.metrics.mean_absolute_error(y_true=unscale(test_y[:340]), y_pred=unscale(test_pred_y)),2)
r1_1t_mae = np.round(sklearn.metrics.mean_absolute_error(y_true=unscale(test_y), y_pred=unscale(data_test_scaled_sisso["r1_1term"])),2)
r1_2t_mae = np.round(sklearn.metrics.mean_absolute_error(y_true=unscale(test_y), y_pred=unscale(data_test_scaled_sisso["r1_2term"])),2)
r2_1t_mae = np.round(sklearn.metrics.mean_absolute_error(y_true=unscale(test_y), y_pred=unscale(data_test_scaled_sisso["r2_1term"])),2)

plt.rcParams["figure.dpi"]=200
fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, sharey=True)
ax1.set_ylabel("DFT Decomposition Energy (eV/Atom)")
ax2.set_xlabel("Predicted Decomposition Energy (eV/Atom)")
 
ax1.scatter(x=unscale(data_test_scaled_sisso["r1_1term"]), y=unscale(test_y), marker="v", color="red",alpha=0.2, label=f"Rung 1, 1-Term, MAE={r1_1t_mae}")
ax2.scatter(x=unscale(data_test_scaled_sisso["r1_2term"]), y=unscale(test_y), marker="^", color="green", alpha=0.2, label=f"Rung 1, 2-Term, MAE={r1_2t_mae}")
ax3.scatter(x=unscale(data_test_scaled_sisso["r2_1term"]), y=unscale(test_y), marker="s", color="blue", alpha=0.2, label=f"Rung 2, 1-term, MAE={r2_1t_mae}")
for ax in (ax1, ax2, ax3):
    ax.scatter(x=unscale(test_pred_y[:340]), y=unscale(test_y[:340]), color="black", alpha=0.3, marker="+", label=f"TPOT, 108 Terms, MAE={tpot_mae}")
    ax.plot([0, 3], [0, 3], color="black", linestyle="--", label="Parity")

ax2.set_title("Testing Set (20% Holdout)")
for ax in (ax1, ax2, ax3):
    ax.legend(prop={"size":5}, loc="upper center")

plt.show()


# In[]:


imps = list(zip(model.feature_importances_, data_train_scaled.drop(columns=[target]).columns))
sorted(imps, key=lambda i: -i[0])


# In[ ]:





# In[ ]:




