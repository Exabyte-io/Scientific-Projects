#!/usr/bin/env python
# coding: utf-8

# In[]:


import math
import pandas as pd

import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import tpot


# In[]:


mean = pd.read_pickle("../dataset_means_stds/bg_mean.pkl")
std = pd.read_pickle("../dataset_means_stds/bg_std.pkl")
data_train_scaled = pd.read_csv("../scaled_featurized_train/scaled_bg_train.csv", index_col=0)
data_test_scaled = pd.read_csv("../scaled_featurized_test/scaled_bg_test.csv", index_col=0)
data_train = (data_train_scaled * std) + mean
data_test = (data_test_scaled * std) + mean


# In[]:


data_train.head()

std


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


target = "bandgap (eV)"
train_x = data_train_scaled.drop(columns=[target]).to_numpy()
train_y = data_train_scaled[target].to_numpy().ravel()

test_x = data_test_scaled.drop(columns=[target]).to_numpy()
test_y = data_test_scaled[target].to_numpy().ravel()

model.fit(train_x, train_y)


# In[]:


def unscale(arr):
    return arr * std[target] + mean[target]
    
train_pred_y = model.predict(train_x)
test_pred_y = model.predict(test_x)


# In[]:


models = {}

models["r1_1term"] = lambda df: 5.482132320909957e-03 + 3.278643371383011e-01 * (df["ave:en_ghosh"] - df["ave:density"])
models["r1_2term"] = lambda df: -4.608150625850920e-01 + 3.272946916567465e-01 * np.exp(-df["ave:density"]) +                                  2.996458319104499e-01 * (df["ave:en_ghosh"] - df["ave:density"])
models["r2_1term"] = lambda df: -4.282838209517619e-01 + 3.031381295978055e-01 * ((df["ave:gs_energy"] + df["ave:en_ghosh"]) * np.exp(-df["ave:density"]))

data_train_scaled_sisso = data_train_scaled.copy()
data_test_scaled_sisso = data_test_scaled.copy()
for key,fun in models.items():
    data_train_scaled_sisso[key] = fun(data_train_scaled_sisso)
    data_test_scaled_sisso[key] = fun(data_test_scaled_sisso)

data_train_scaled_sisso.head()


# In[]:


tpot_mape =  np.round(sklearn.metrics.mean_absolute_error(y_true=unscale(train_y), y_pred=unscale(train_pred_y)),2)
r1_1t_mape = np.round(sklearn.metrics.mean_absolute_error(y_true=unscale(train_y), y_pred=unscale(data_train_scaled_sisso["r1_1term"])),2)
r1_2t_mape = np.round(sklearn.metrics.mean_absolute_error(y_true=unscale(train_y), y_pred=unscale(data_train_scaled_sisso["r1_2term"])),2)
r2_1t_mape = np.round(sklearn.metrics.mean_absolute_error(y_true=unscale(train_y), y_pred=unscale(data_train_scaled_sisso["r2_1term"])),2)

plt.rcParams["figure.dpi"]=200
fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, sharey=True)
ax1.set_ylabel("DFT Bandgap (eV)")
ax2.set_xlabel("Predicted Bandgap (eV)")


    
ax1.scatter(x=unscale(data_train_scaled_sisso["r1_1term"]), y=unscale(train_y), marker="v", color="red",alpha=0.2, label=f"Rung 1, 1-Term, MAE={r1_1t_mape}")
ax2.scatter(x=unscale(data_train_scaled_sisso["r1_2term"]), y=unscale(train_y), marker="^", color="green", alpha=0.2, label=f"Rung 1, 2-Term, MAE={r1_2t_mape}")
ax3.scatter(x=unscale(data_train_scaled_sisso["r2_1term"]), y=unscale(train_y), marker="s", color="blue", alpha=0.2, label=f"Rung 2, 1-term, MAE={r2_1t_mape}")
for ax in (ax1, ax2, ax3):
    ax.scatter(x=unscale(train_pred_y), y=unscale(train_y), color="black", alpha=0.3, marker="+", label=f"TPOT, 108 Terms, MAE={tpot_mape}")
    ax.plot([0, 8], [-0, 8], color="black", linestyle="--", label="Parity")
    
for ax in (ax1, ax2, ax3):
    ax.legend(prop={"size":5}, loc="upper center")

ax2.set_title("Training Set (80% of Dataset)")
plt.show()


# In[]:


tpot_mape =  np.round(sklearn.metrics.mean_absolute_error(y_true=unscale(test_y), y_pred=unscale(test_pred_y)),2)
r1_1t_mape = np.round(sklearn.metrics.mean_absolute_error(y_true=unscale(test_y), y_pred=unscale(data_test_scaled_sisso["r1_1term"])),2)
r1_2t_mape = np.round(sklearn.metrics.mean_absolute_error(y_true=unscale(test_y), y_pred=unscale(data_test_scaled_sisso["r1_2term"])),2)
r2_1t_mape = np.round(sklearn.metrics.mean_absolute_error(y_true=unscale(test_y), y_pred=unscale(data_test_scaled_sisso["r2_1term"])),2)

plt.rcParams["figure.dpi"]=200
fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, sharey=True)
ax1.set_ylabel("DFT Bandgap (eV)")
ax2.set_xlabel("Predicted Bandgap (eV)")
 
ax1.scatter(x=unscale(data_test_scaled_sisso["r1_1term"]), y=unscale(test_y), marker="v", color="red",alpha=0.2, label=f"Rung 1, 1-Term, MAE={r1_1t_mape}")
ax2.scatter(x=unscale(data_test_scaled_sisso["r1_2term"]), y=unscale(test_y), marker="^", color="green", alpha=0.2, label=f"Rung 1, 2-Term, MAE={r1_2t_mape}")
ax3.scatter(x=unscale(data_test_scaled_sisso["r2_1term"]), y=unscale(test_y), marker="s", color="blue", alpha=0.2, label=f"Rung 2, 1-term, MAE={r2_1t_mape}")
for ax in (ax1, ax2, ax3):
    ax.scatter(x=unscale(test_pred_y), y=unscale(test_y), color="black", alpha=0.3, marker="+", label=f"TPOT, 108 Terms, MAE={tpot_mape}")
    ax.plot([0, 8], [0, 8], color="black", linestyle="--", label="Parity")

ax2.set_title("Testing Set (20% Holdout)")
for ax in (ax1, ax2, ax3):
    ax.legend(prop={"size":5}, loc="upper center")

plt.show()


# In[]:


model.fitted_pipeline_


# In[]:


sum(model.fitted_pipeline_[4].get_support())


# # model.fitted_pipeline_[3].__dict__

# In[ ]:




