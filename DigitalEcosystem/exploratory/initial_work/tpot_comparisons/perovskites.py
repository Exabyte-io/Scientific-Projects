#!/usr/bin/env python
# coding: utf-8

# In[]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tpot
import sklearn


# In[]:


mean = pd.read_pickle("../dataset_means_stds/perov_mean.pkl")
std = pd.read_pickle("../dataset_means_stds/perov_std.pkl")
data_train_scaled = pd.read_csv("../scaled_featurized_train/scaled_perovskite_train.csv", index_col=0)
data_test_scaled = pd.read_csv("../scaled_featurized_test/scaled_perovskite_test.csv", index_col=0)
data_train = (data_train_scaled * std) + mean
data_test = (data_test_scaled * std) + mean


# In[ ]:


# 1-Rung features from not scaling the volumes. Probably gonna delete later, but keeping these around for now rather than doing a git revert later.
# features = {}
# features["r1f1"] = lambda df: df["ave:vdw_radius_uff"] + df["ave:gs_est_fcc_latcnt"]
# features["r1f2"] = lambda df: abs(df["ave:sound_velocity"] - df["ave:specific_heat"])
# features["r1f3"] = lambda df: df["ave:period"] - df["ave:bulk_modulus"]
# features["r1f4"] = lambda df: df["ave:thermal_conductivity"] * df["ave:heat_capacity_mass"]
# features["r1f5"] = lambda df: df["ave:hhi_r"] / df["ave:gs_energy"]
# features["r1f6"] = lambda df: df["ave:vdw_radius_alvarez"] * df["ave:num_d_unfilled"]
# features["r1f7"] = lambda df: df["ave:num_d_unfilled"] * df["ave:gs_est_bcc_latcnt"]
# features["r1f8"] = lambda df: df["ave:c6_gb"] / df["ave:num_f_valence"]
# features["r1f9"] = lambda df: df["ave:num_s_valence"] * df["ave:en_ghosh"]
# features["r1fA"] = lambda df: df["ave:heat_capacity_molar"] * df["ave:electron_affinity"]
# features["r1fB"] = lambda df: df["ave:gs_est_bcc_latcnt"] + df["ave:atomic_number"]
# features["r2f1"] = lambda df: abs(df["ave:specific_heat"] - df["ave:covalent_radius_pyykko_double"]) * (df["ave:num_p_valence"] - df["ave:electron_negativity"])
# features["r2f2"] = lambda df: abs((df["ave:icsd_volume"] * df["ave:atomic_volume"]) - (df["ave:vdw_radius_uff"] + df["ave:boiling_point"]))
# features["r2f3"] = lambda df: abs(df["ave:specific_heat"] - df["ave:covalent_radius_pyykko_double"]) * (df["ave:num_p_valence"] - df["ave:en_pauling"])
# features["r2f4"] = lambda df: ((df["ave:heat_capacity_molar"] / df["ave:atomic_weight"]) / (df["ave:num_d_unfilled"] - df["ave:heat_of_formation"]))
# features["r2f5"] = lambda df: ((df["ave:sound_velocity"] + df["ave:num_p_unfilled"]) * (df["ave:melting_point"] + df["ave:electron_affinity"]))
# features["r2f6"] = lambda df: abs(abs(df["ave:atomic_weight"] - df["ave:atomic_radius_rahm"]) - (df["ave:thermal_conductivity"] + df["ave:num_unfilled"]))
# features["r2f7"] = lambda df: ((df["ave:num_valance"]**3) / (df["ave:specific_heat"] + df["ave:melting_point"])) 
# features["r2f8"] = lambda df: ((df["ave:hhi_r"] + df["ave:en_pauling"]) * (df["ave:mendeleev_number"] + df["ave:c6_gb"]))
# features["r2f9"] = lambda df:  abs(abs(df["ave:covalent_radius_pyykko_triple"] - df["ave:atomic_number"]) - (df["ave:thermal_conductivity"] + df["ave:num_unfilled"]))


# data_train_scaled_sisso = data_train_scaled.copy()
# data_test_scaled_sisso = data_test_scaled.copy()
# for key,fun in features.items():
#     data_train_scaled_sisso[key] = fun(data_train_scaled_sisso)
#     data_test_scaled_sisso[key] = fun(data_test_scaled_sisso)

# data_train_scaled_sisso.head()


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




