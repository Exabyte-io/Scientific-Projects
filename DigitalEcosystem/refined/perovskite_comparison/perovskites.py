#!/usr/bin/env python
# coding: utf-8

# # Perovskites
# 
# In this notebook, we demonstrate some of the initial work on this project that was carried out in the summer of 2021. Here, we leverage some Xenonpy-calculated properties to calculate the unit cell volume of several perovskite systems taken from NOMAD's database.
# 
# Both SISSO and TPOT are used t

# In[]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tpot
import sklearn

from IPython.display import Latex


# # Read in the Data
# 
# At this point, we hadn't yet settled on how we would be pipelineing the data. We were storing the mean and standard deviation of the dataset separately in pickled pandas dataframes. So, we'll load in the mean and standard deviation, along with the training / testing set.

# In[]:


mean = pd.read_pickle("./dataset_means_stds/perov_mean.pkl")
std = pd.read_pickle("./dataset_means_stds/perov_std.pkl")
data_train_scaled = pd.read_csv("./scaled_featurized_train/scaled_perovskite_train.csv", index_col=0)
data_test_scaled = pd.read_csv("./scaled_featurized_test/scaled_perovskite_test.csv", index_col=0)
data_train = (data_train_scaled * std) + mean
data_test = (data_test_scaled * std) + mean


# # SISSO Models
# Here, we have Rung1 and Rung2 models that have been generated by SISSO using the XenonPy compositional descriptors. In all casees, $V$ refers to Volume. We have also explained the features generated by XenonPy as described on their documentation page. [Link to Documentation].(https://xenonpy.readthedocs.io/en/latest/features.html)
# 
# ## Rung1, 1-Term
# $V=c_0+a_0*(VDW\_Radius\_UFF_{avg} + Covalent\_Radius\_Pyykko\_Double_{avg})$
# 
# Where:
# - $VDW\_Radius\_UFF_{avg}$ is average the Van der Waals radius used by the Universal ForceField (UFF) method.
# - $Covalent\_Radius\_Pyykko\_Double_{avg}$ is the average double-bond covalent radius by Pyykko et al.
# 
# ## Rung1, 2-Term
# $V=c_0+a_0*(Num\_P\_Valence_{avg} + Num\_P\_Unfilled_{avg}) + a_1*(GS\_Volume\_Per_{min} + Covalent\_Radius\_Pyykko\_Double_{avg})$
# 
# Where:
# - $Num\_P\_Valence_{avg}$ is the average number of valence electrons in P orbitals.
# - $Num\_P\_Unfilled_{avg}$ is the average number of unfilled valence electrons in the P orbitals.
# - $GS\_Volume\_Per_{min}$ is the minimum DFT-calculated volume of the elemental unit cells.
# - $Covalent\_Radius\_Pyykko\_Double_{avg}$ is the average double-bond covalent radius by Pyykko et al.
# 
# ## Rung2, 1-Term
# $V=c_0+a_0*((Polarizability_{min}+Num\_P\_Valence_{avg})+(GS\_Volume\_Per_{min}+VDW\_Radius\_MM3_{avg}))$
# 
# Where:
# - $Polarizability_{min}$ is the minimum polarizability.
# - $Num\_P\_Valence_{avg}$ is the average number of valence electrons in P orbitals.
# - $GS\_Volume\_Per_{min}$ is the minimum DFT-calculated volume of the elemental unit cells.
# - $VDW\_Radius\_M3_{avg}$ is the average Van der Waals radius used by the MM3 forcefield.
# 

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


# # TPOT Model
# 
# To have something to compare to, we'll also run a TPOT model for 10 minutes.

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


# ## TPOT Pipeline
# TPOT Generates the following pipeline:
# 1. Decision Tree Regressor
# 2. The decision tree's predictions are scaled to be between 0 and 1.
# 3. Ridge regression is used on this scaled set of predictions.
# 
# ## TPOT Model Plot

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


# # SISSO Rung1, 1-Term Plot

# In[]:


plt.scatter(x=unscale(data_train_scaled_sisso["r1_1term"]), y=unscale(train_y), label="Train")
plt.scatter(x=unscale(data_test_scaled_sisso["r1_1term"]), y=unscale(test_y), label="Test")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.legend()


# # SISSO Rung1, 2-Term Plot

# In[]:


plt.scatter(x=unscale(data_train_scaled_sisso["r1_2term"]), y=unscale(train_y), label="Train")
plt.scatter(x=unscale(data_test_scaled_sisso["r1_2term"]), y=unscale(test_y), label="Test")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.legend()


# # SISSO Rung2, 1-Term Plot

# In[]:


plt.scatter(x=unscale(data_train_scaled_sisso["r2_1term"]), y=unscale(train_y), label="Train")
plt.scatter(x=unscale(data_test_scaled_sisso["r2_1term"]), y=unscale(test_y), label="Test")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.legend()


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


# ## Test-Set
# Below, we plot the Training set results for:
# 1. The TPOT model (Black + symbols)
# 2. The SISSO Rung1, 1-Term Model (Red Nablas / Upside-Down Triangles)
# 3. The SISSO Rung1, 2-Term Model (Green Deltas / Rightside-Up Triangles)
# 4. The SISSO Rung2, 1-Term Model (Blue Squares)
# 
# As a guide to the eye, parity is also drawn as a dashed black line.

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




