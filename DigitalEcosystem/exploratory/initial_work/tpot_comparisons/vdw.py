#!/usr/bin/env python
# coding: utf-8

# In[]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tpot
import sklearn


# In[]:


mean = pd.read_pickle("../dataset_means_stds/vdw_mean.pkl")
std = pd.read_pickle("../dataset_means_stds/vdw_std.pkl")
data_train_scaled = pd.read_csv("../scaled_featurized_train/scaled_vdw_train.csv", index_col=0).dropna(axis=1)
data_test_scaled = pd.read_csv("../scaled_featurized_test/scaled_vdw_test.csv", index_col=0).dropna(axis=1)
data_train = (data_train_scaled * std) + mean
data_test = (data_test_scaled * std) + mean

data_train.head()


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


target = "energy_vdw_per_atom (eV/atom)"
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

models["r1_1term"] = lambda df: 9.027012811387233e-04 + 5.652128235876568e-01 * (df["ave:gs_energy"] - df["ave:en_ghosh"])
models["r1_2term"] = lambda df: 1.801668309432753e-03 + 3.416811904296799e-01 * (df["ave:first_ion_en"]  + df["ave:density"]) +                                 6.772451781519770e-01 * (df["ave:gs_energy"] - df["ave:en_ghosh"])
models["r2_1term"] = lambda df: -8.613014913447112e-01 + 6.775861689045741e-01 * ((df["ave:gs_energy"]  - df["ave:en_ghosh"]) + np.exp(-df["ave:c6_gb"]))

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
ax1.set_ylabel("Actual DFT VDW Energy (eV)")
ax2.set_xlabel("Predicted DFT VDW Energy (eV)")


    
ax1.scatter(x=unscale(data_train_scaled_sisso["r1_1term"]), y=unscale(train_y), marker="v", color="red",alpha=0.2, label=f"Rung 1, 1-Term, MAE={r1_1t_mae}")
ax2.scatter(x=unscale(data_train_scaled_sisso["r1_2term"]), y=unscale(train_y), marker="^", color="green", alpha=0.2, label=f"Rung 1, 2-Term, MAE={r1_2t_mae}")
ax3.scatter(x=unscale(data_train_scaled_sisso["r2_1term"]), y=unscale(train_y), marker="s", color="blue", alpha=0.2, label=f"Rung 2, 1-term, MAE={r2_1t_mae}")
for ax in (ax1, ax2, ax3):
    ax.scatter(x=unscale(train_pred_y), y=unscale(train_y), color="black", alpha=0.3, marker="+", label=f"TPOT, 77 Terms, MAE={tpot_mae}")
    ax.plot([0, 3], [0, 3], color="black", linestyle="--", label="Parity")
    
for ax in (ax1, ax2, ax3):
    ax.legend(prop={"size":5}, loc="upper center")

ax2.set_title("Training Set (80% of Dataset)")
plt.show()


# In[]:


tpot_mae =  np.round(sklearn.metrics.mean_absolute_error(y_true=unscale(test_y), y_pred=unscale(test_pred_y)),2)
r1_1t_mae = np.round(sklearn.metrics.mean_absolute_error(y_true=unscale(test_y), y_pred=unscale(data_test_scaled_sisso["r1_1term"])),2)
r1_2t_mae = np.round(sklearn.metrics.mean_absolute_error(y_true=unscale(test_y), y_pred=unscale(data_test_scaled_sisso["r1_2term"])),2)
r2_1t_mae = np.round(sklearn.metrics.mean_absolute_error(y_true=unscale(test_y), y_pred=unscale(data_test_scaled_sisso["r2_1term"])),2)

plt.rcParams["figure.dpi"]=200
fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, sharey=True)
ax1.set_ylabel("DFT VDW Energy (eV/Atom)")
ax2.set_xlabel("Predicted DFT VDW Energy (eV/Atom)")
 
ax1.scatter(x=unscale(data_test_scaled_sisso["r1_1term"]), y=unscale(test_y), marker="v", color="red",alpha=0.2, label=f"Rung 1, 1-Term, MAE={r1_1t_mae}")
ax2.scatter(x=unscale(data_test_scaled_sisso["r1_2term"]), y=unscale(test_y), marker="^", color="green", alpha=0.2, label=f"Rung 1, 2-Term, MAE={r1_2t_mae}")
ax3.scatter(x=unscale(data_test_scaled_sisso["r2_1term"]), y=unscale(test_y), marker="s", color="blue", alpha=0.2, label=f"Rung 2, 1-term, MAE={r2_1t_mae}")
for ax in (ax1, ax2, ax3):
    ax.scatter(x=unscale(test_pred_y), y=unscale(test_y), color="black", alpha=0.3, marker="+", label=f"TPOT, 77 Terms, MAE={tpot_mae}")
    ax.plot([0, 3], [0, 3], color="black", linestyle="--", label="Parity")

ax2.set_title("Testing Set (20% Holdout)")
for ax in (ax1, ax2, ax3):
    ax.legend(prop={"size":5}, loc="upper center")

plt.show()


# In[]:


data_train.shape


# ###### 
