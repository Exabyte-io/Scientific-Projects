#!/usr/bin/env python
# coding: utf-8

# In[]:


import functools
import random

import pandas as pd
import numpy as np

import sklearn.preprocessing
import sklearn.feature_selection
import sklearn.pipeline

import ase

from dscribe.descriptors import SineMatrix


# In[]:


# Read the data
datafile = "../raw_data/2d_mat_dataset_raw.pkl"
data = pd.read_pickle(datafile)
data = data[data["discovery_process (unitless)"] == "top-down"]
initial_size = len(data)

data.head()


# In[]:


target_cols = ["decomposition_energy (eV/atom)", "exfoliation_energy_per_atom (eV/atom)", "bandgap (eV)"]
def should_keep_col(col):
    if "ave" not in col:
        return False
    
    if "num" in col:
        return False
    
    radius_to_keep= "atomic_radius"
    if "radius" in col and col != radius_to_keep:
        return False
    
    return True
average_cols = data.columns[[True if should_keep_col(i) else False for i in data.columns]]
separated_atoms_col = "sum:gs_energy"
atoms_obj_col = "atoms_object (unitless)"

new_data = data[target_cols + [atoms_obj_col] + list(average_cols) + [separated_atoms_col]].dropna().reset_index().drop(columns=["index"])
new_data.head()


# In[]:


largest_system =  new_data["atoms_object (unitless)"].apply(len).max()
sm = SineMatrix(
    n_atoms_max = largest_system,
    permutation = "eigenspectrum",
    sparse = False,
    flatten = True
)

def get_sm(atoms):
    new_cols = sm.create(atoms).reshape(1,-1).flatten()
    return new_cols

raw_sines = new_data["atoms_object (unitless)"].apply(get_sm)
refined_sines = np.vstack(raw_sines)

# This results on some very small (e.g. 10^-14) imaginary components. We'll remove those.
refined_sines = np.real(refined_sines)

sine_df = pd.DataFrame(refined_sines, columns=[f"sine_eigenspectrum_{i}" for i in range(sm.n_atoms_max)])
sine_df.head()


# In[]:


featurized_data = pd.concat([new_data, sine_df], axis=1).drop(columns=["atoms_object (unitless)"])
featurized_data


# In[]:


# Get datasets
decomp_target = "decomposition_energy (eV/atom)"
exfol_target = "exfoliation_energy_per_atom (eV/atom)"
bg_target = "bandgap (eV)"

descriptor_cols = list(featurized_data.columns[3:])
def get_dataset_subset(target_column, feature_columns, full_dataset):
    # Generate Dataset
    initial_size = len(full_dataset)
    result_data = full_dataset[[target_column] + feature_columns].dropna()
    print(f"Dropped {initial_size - len(result_data)} missing rows for target {target_column}")
    return result_data

data_extractor = functools.partial(get_dataset_subset, feature_columns=descriptor_cols, full_dataset=featurized_data)

decomp_data = data_extractor(decomp_target)
exfol_data = data_extractor(exfol_target)
bg_data = data_extractor(bg_target)


# In[]:


# Train/Test Split
np.random.seed(1234)
random.seed(1234)

decomp_train, decomp_test = sklearn.model_selection.train_test_split(decomp_data, test_size=0.2)
exfol_train, exfol_test = sklearn.model_selection.train_test_split(exfol_data, test_size=0.2)
bg_train, bg_test = sklearn.model_selection.train_test_split(bg_data, test_size=0.2)


# In[]:


dataset_mean_path = "dataset_means_stds"
# Scale the dataset
def z_score_scale(dataset, mean=None, std=None):
    if mean is None:
        mean = dataset.mean()
    if std is None:
        std = dataset.std()
    
    result = (dataset - mean) / std
    
    return result.copy(), mean, std

decomp_scaled, decomp_mean, decomp_std = z_score_scale(decomp_train)
decomp_mean.to_pickle(f"{dataset_mean_path}/topdown_decomp_mean.pkl")
decomp_mean.to_pickle(f"{dataset_mean_path}/topdown_decomp_std.pkl")

exfol_scaled, exfol_mean, exfol_std = z_score_scale(exfol_train)
exfol_mean.to_pickle(f"{dataset_mean_path}/topdown_exfol_mean.pkl")
exfol_std.to_pickle(f"{dataset_mean_path}/topdown_exfol_std.pkl")

bg_scaled, bg_mean, bg_std = z_score_scale(bg_train)
bg_mean.to_pickle(f"{dataset_mean_path}/topdown_bg_mean.pkl")
bg_std.to_pickle(f"{dataset_mean_path}/topdown_bg_std.pkl")


# In[]:


# Write to CSV
decomp_scaled.to_csv("scaled_featurized_train/scaled_topdown_decomp_train.csv")
exfol_scaled.to_csv("scaled_featurized_train/scaled_topdown_exfol_train.csv")
bg_scaled.to_csv("scaled_featurized_train/scaled_topdown_bg_train.csv")


# In[]:


# Scale the test set

decomp_test_scaled, _, _ = z_score_scale(decomp_test, decomp_mean, decomp_std)
decomp_test_scaled.to_csv("scaled_featurized_test/scaled_topdown_decomp_test.csv")

exfol_test_scaled, _, _ = z_score_scale(exfol_test, exfol_mean, exfol_std)
exfol_test_scaled.to_csv("scaled_featurized_test/scaled_topdown_exfol_test.csv")

bg_test_scaled, _, _ = z_score_scale(bg_test, bg_mean, bg_std)
bg_test_scaled.to_csv("scaled_featurized_test/scaled_topdown_bg_test.csv")


# In[ ]:




