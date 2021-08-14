#!/usr/bin/env python
# coding: utf-8

# In[]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tpot
import sklearn


# In[]:


mean = pd.read_pickle("./dataset_means_stds/perov_mean.pkl")
std = pd.read_pickle("./dataset_means_stds/perov_std.pkl")
data_train_scaled = pd.read_csv("./scaled_featurized_train/scaled_perovskite_train.csv", index_col=0)
data_test_scaled = pd.read_csv("./scaled_featurized_test/scaled_perovskite_test.csv", index_col=0)
data_train = (data_train_scaled * std) + mean
data_test = (data_test_scaled * std) + mean


# In[]:


models = {}

models["r1_1term"] = lambda df: 2.342082515585748e-02 + 5.172456498122173e-01 * (df["ave:vdw_radius_uff"] + df["ave:covalent_radius_pyykko_double"])
models["r1_2term"] = lambda df: 1.052443572291616e-02 + 1.242091145866421e-01 * (df["ave:num_p_valence"] + df["ave:num_p_unfilled"]) +                                 4.898720001428966e-01 * (df["min:gs_volume_per"] + df["ave:covalent_radius_pyykko_double"])
models["r2_1term"] = lambda df: 4.521699008847579e-03 + 2.966651096729857e-01 * ((df["min:Polarizability"] + df["ave:num_p_valence"]) + (df["min:gs_volume_per"] + df["ave:vdw_radius_mm3"]))

data_train_scaled_sisso = data_train_scaled.copy()
data_test_scaled_sisso = data_test_scaled.copy()
for key,fun in models.items():
    data_train_scaled_sisso[key] = fun(data_train_scaled_sisso)
    data_test_scaled_sisso[key] = fun(data_test_scaled_sisso)

data_train_scaled_sisso.head()


# In[]:


# TPOT with the vanilla features

model = tpot.TPOTRegressor(
    generations=None,
    population_size=100,
    max_eval_time_mins=1/60,
    max_time_mins=10,
    cv=len(data_train_scaled),
    verbosity=2,
    scoring="neg_root_mean_squared_error",
    config_dict=tpot.config.regressor_config_dict,
    n_jobs=-1,
    random_state=1234
)


# In[]:


train_x = data_train_scaled.drop(columns="Volume").to_numpy()
train_y = data_train_scaled.Volume.to_numpy().ravel()

test_x = data_test_scaled.drop(columns="Volume").to_numpy()
test_y = data_test_scaled.Volume.to_numpy().ravel()

model.fit(train_x, train_y)


# In[]:


def unscale(arr):
    return arr * std["Volume"] + mean["Volume"]
    
train_pred_y = model.predict(train_x) 
test_pred_y = model.predict(test_x)

plt.scatter(x=unscale(train_pred_y), y=unscale(train_y), label="Train")
plt.scatter(x=unscale(test_pred_y), y=unscale(test_y), label="Test")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.legend()


# In[]:


plt.scatter(x=unscale(data_train_scaled_sisso["r1_1term"]), y=unscale(train_y), label="Train")
plt.scatter(x=unscale(data_test_scaled_sisso["r1_1term"]), y=unscale(test_y), label="Test")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.legend()


# In[]:


plt.scatter(x=unscale(data_train_scaled_sisso["r1_2term"]), y=unscale(train_y), label="Train")
plt.scatter(x=unscale(data_test_scaled_sisso["r1_2term"]), y=unscale(test_y), label="Test")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.legend()


# In[]:


plt.scatter(x=unscale(data_train_scaled_sisso["r2_1term"]), y=unscale(train_y), label="Train")
plt.scatter(x=unscale(data_test_scaled_sisso["r2_1term"]), y=unscale(test_y), label="Test")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.legend()


# In[]:


tpot_mape = np.round(sklearn.metrics.mean_absolute_percentage_error(y_true=unscale(train_y), y_pred=unscale(train_pred_y)),2)
r1_1t_mape = np.round(sklearn.metrics.mean_absolute_percentage_error(y_true=unscale(train_y), y_pred=unscale(data_train_scaled_sisso["r1_1term"])),2)
r1_2t_mape = np.round(sklearn.metrics.mean_absolute_percentage_error(y_true=unscale(train_y), y_pred=unscale(data_train_scaled_sisso["r1_2term"])),2)
r2_1t_mape = np.round(sklearn.metrics.mean_absolute_percentage_error(y_true=unscale(train_y), y_pred=unscale(data_train_scaled_sisso["r2_1term"])),2)

plt.rcParams["figure.dpi"]=200
plt.scatter(x=unscale(train_pred_y), y=unscale(train_y), color="black", alpha=0.9, marker="+", label=f"TPOT, 108 Terms, MAPE={tpot_mape}")
plt.scatter(x=unscale(data_train_scaled_sisso["r1_1term"]), y=unscale(train_y), marker="v", color="red",alpha=0.5, label=f"Rung 1, 1-Term, MAPE={r1_1t_mape}")
plt.scatter(x=unscale(data_train_scaled_sisso["r1_2term"]), y=unscale(train_y), marker="^", color="green", alpha=0.5, label=f"Rung 1, 2-Term, MAPE={r1_2t_mape}")
plt.scatter(x=unscale(data_train_scaled_sisso["r2_1term"]), y=unscale(train_y), marker="s", color="blue", alpha=0.5, label=f"Rung 2, 1-term, MAPE={r2_1t_mape}")
plt.plot([45, 280], [45, 280], color="black", linestyle="--", label="Parity")

plt.title("Training Set (80% of Dataset)")
plt.xlabel("Predicted (Å^3 / Formula Unit)")
plt.ylabel("Actual Volume (Å^3 / Formula Unit)")
plt.legend(prop={"size": 8})
plt.show()


# In[]:


tpot_mape = np.round(sklearn.metrics.mean_absolute_percentage_error(y_true=unscale(test_y), y_pred=unscale(test_pred_y)),2)
r1_1t_mape = np.round(sklearn.metrics.mean_absolute_percentage_error(y_true=unscale(test_y), y_pred=unscale(data_test_scaled_sisso["r1_1term"])),2)
r1_2t_mape = np.round(sklearn.metrics.mean_absolute_percentage_error(y_true=unscale(test_y), y_pred=unscale(data_test_scaled_sisso["r1_2term"])),2)
r2_1t_mape = np.round(sklearn.metrics.mean_absolute_percentage_error(y_true=unscale(test_y), y_pred=unscale(data_test_scaled_sisso["r2_1term"])),2)

plt.rcParams["figure.dpi"]=200
plt.scatter(x=unscale(test_pred_y), y=unscale(test_y), color="black", alpha=0.9, marker="+", label=f"TPOT, 108 Terms, MAPE={tpot_mape}")
plt.scatter(x=unscale(data_test_scaled_sisso["r1_1term"]), y=unscale(test_y), marker="v", color="red",alpha=0.5, label=f"Rung 1, 1-Term, MAPE={r1_1t_mape}")
plt.scatter(x=unscale(data_test_scaled_sisso["r1_2term"]), y=unscale(test_y), marker="^", color="green", alpha=0.5, label=f"Rung 1, 2-Term, MAPE={r1_2t_mape}")
plt.scatter(x=unscale(data_test_scaled_sisso["r2_1term"]), y=unscale(test_y), marker="s", color="blue", alpha=0.5, label=f"Rung 2, 1-term, MAPE={r2_1t_mape}")
plt.plot([45, 280], [45, 280], color="black", linestyle="--", label="Parity")

plt.title("Testing Set (20% Holdout)")
plt.xlabel("Predicted (Å^3 / Formula Unit)")
plt.ylabel("Actual Volume (Å^3 / Formula Unit)")
plt.legend(prop={"size": 8})
plt.show()


# In[]:


model.fitted_pipeline_[0].__dict__['estimator'].__dict__


# In[ ]:




