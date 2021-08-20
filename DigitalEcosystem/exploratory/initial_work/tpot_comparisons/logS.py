#!/usr/bin/env python
# coding: utf-8

# In[]:


import math

import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import tpot


# In[]:


mean = pd.read_pickle("../dataset_means_stds/logS_mean.pkl")
std = pd.read_pickle("../dataset_means_stds/logS_std.pkl")
data_train_scaled = pd.read_csv("../scaled_featurized_train/scaled_logS_train.csv", index_col=0)
data_test_scaled = pd.read_csv("../scaled_featurized_test/scaled_logS_test.csv", index_col=0)
data_train = (data_train_scaled * std) + mean
data_test = (data_test_scaled * std) + mean


# In[]:


# features = {}
# features["r1f1"] = lambda df: df["MolLogP"].apply(math.sin)
# features["r1f2"] = lambda df: df["MolLogP"] + df["MolWt"]
# features["r1f3"] = lambda df: df["BalabanJ"] * df["MolMR"]
# features["r1f4"] = lambda df: df["BertzCT"] + df["MolLogP"]
# features["r1f5"] = lambda df: df["NumValenceElectrons"] - df["HeavyAtomCount"]
# features["r1f6"] = lambda df: df["BertzCT"] * df["BalabanJ"]
# features["r1f7"] = lambda df: abs(df["TPSA"] - df["MolLogP"])
# features["r1f8"] = lambda df: df["BertzCT"] - df["NumValenceElectrons"]
# features["r1f9"] = lambda df: abs(df["MolWt"])

# data_train_scaled_sisso = data_train_scaled.copy()
# data_test_scaled_sisso = data_test_scaled.copy()
# for key,fun in features.items():
#     data_train_scaled_sisso[key] = fun(data_train_scaled_sisso)
#     data_test_scaled_sisso[key] = fun(data_test_scaled_sisso)

# data_train_scaled_sisso.head()


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


train_x = data_train_scaled.drop(columns="Solubility").to_numpy()
train_y = data_train_scaled.Solubility.to_numpy().ravel()

test_x = data_test_scaled.drop(columns="Solubility").to_numpy()
test_y = data_test_scaled.Solubility.to_numpy().ravel()

model.fit(train_x, train_y)


# In[]:


def unscale(arr):
    return arr * std["Solubility"] + mean["Solubility"]
    
train_pred_y = model.predict(train_x)
test_pred_y = model.predict(test_x)


# In[]:


models = {}

models["r1_1term"] = lambda df: 1.313876535718961e-02 + -1.339440350284874e+00 * df["MolLogP"].apply(math.sin)
models["r1_2term"] = lambda df: 9.901970108755258e-03 + -2.054433417991656e-01 * (df["MolLogP"] + df["MolWt"]) +                                 -9.837466166023564e-01 * df["MolLogP"].apply(math.sin)
models["r2_1term"] = lambda df:  6.673682610827048e-02 + -1.115267690836167e+00 * ((df["MolLogP"] / df["BalabanJ"]) * df["BalabanJ"].apply(math.sin))

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
ax1.set_ylabel("Actual LogS")
ax2.set_xlabel("Predicted LogS")


    
ax1.scatter(x=unscale(data_train_scaled_sisso["r1_1term"]), y=unscale(train_y), marker="v", color="red",alpha=0.1, label=f"Rung 1, 1-Term, MAE={r1_1t_mape}")
ax2.scatter(x=unscale(data_train_scaled_sisso["r1_2term"]), y=unscale(train_y), marker="^", color="green", alpha=0.1, label=f"Rung 1, 2-Term, MAE={r1_2t_mape}")
ax3.scatter(x=unscale(data_train_scaled_sisso["r2_1term"]), y=unscale(train_y), marker="s", color="blue", alpha=0.1, label=f"Rung 2, 1-term, MAE={r2_1t_mape}")
for ax in (ax1, ax2, ax3):
    ax.scatter(x=unscale(train_pred_y), y=unscale(train_y), color="black", alpha=0.3, marker="+", label=f"TPOT, 108 Terms, MAE={tpot_mape}")
    ax.plot([-20, 5], [-20, 5], color="black", linestyle="--", label="Parity")
    
for ax in (ax1, ax2, ax3):
    ax.legend(prop={"size":5}, loc="lower center")

ax2.set_title("Training Set (80% of Dataset)")
plt.show()


# In[]:


tpot_mape =  np.round(sklearn.metrics.mean_absolute_error(y_true=unscale(test_y), y_pred=unscale(test_pred_y)),2)
r1_1t_mape = np.round(sklearn.metrics.mean_absolute_error(y_true=unscale(test_y), y_pred=unscale(data_test_scaled_sisso["r1_1term"])),2)
r1_2t_mape = np.round(sklearn.metrics.mean_absolute_error(y_true=unscale(test_y), y_pred=unscale(data_test_scaled_sisso["r1_2term"])),2)
r2_1t_mape = np.round(sklearn.metrics.mean_absolute_error(y_true=unscale(test_y), y_pred=unscale(data_test_scaled_sisso["r2_1term"])),2)

plt.rcParams["figure.dpi"]=200
fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, sharey=True)
ax1.set_ylabel("Actual LogS")
ax2.set_xlabel("Predicted LogS")
 
ax1.scatter(x=unscale(data_test_scaled_sisso["r1_1term"]), y=unscale(test_y), marker="v", color="red",alpha=0.1, label=f"Rung 1, 1-Term, MAE={r1_1t_mape}")
ax2.scatter(x=unscale(data_test_scaled_sisso["r1_2term"]), y=unscale(test_y), marker="^", color="green", alpha=0.1, label=f"Rung 1, 2-Term, MAE={r1_2t_mape}")
ax3.scatter(x=unscale(data_test_scaled_sisso["r2_1term"]), y=unscale(test_y), marker="s", color="blue", alpha=0.1, label=f"Rung 2, 1-term, MAE={r2_1t_mape}")
for ax in (ax1, ax2, ax3):
    ax.scatter(x=unscale(test_pred_y), y=unscale(test_y), color="black", alpha=0.2, marker="+", label=f"TPOT, 108 Terms, MAE={tpot_mape}")
    ax.plot([-20, 5], [-20, 5], color="black", linestyle="--", label="Parity")

ax2.set_title("Testing Set (20% Holdout)")
for ax in (ax1, ax2, ax3):
    ax.legend(prop={"size":5}, loc="lower center")

plt.show()


# In[]:


model.fitted_pipeline_[0].get_support()


# In[ ]:




