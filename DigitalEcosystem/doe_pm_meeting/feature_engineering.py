#!/usr/bin/env python
# coding: utf-8

# In[]:


import pandas as pd
import numpy as np

import sklearn.preprocessing
import sklearn.feature_selection
import sklearn.pipeline
import matplotlib.pyplot as plt
import ase
import xgboost

from dscribe.descriptors import SineMatrix


# In[]:


# Read the data
datafile = "/Users/mat3ra/sisso_collab/DigitalEcosystem/data/2d_mat_dataset_raw.pkl"
data = pd.read_pickle(datafile)
initial_size = len(data)
data.head()

data.describe()


# # Ideas for Descriptors
# - Similar to the BCM, some measure of how under-coordinated the atoms are relative to their bulk versions
#     - Might be harder for things like Oxygen, for-which we could just use number of covalent bonds or something
# - Keep the weighted averages, they might be useful
# - If we're looking at decomposition energies, what about the energy of the constituent elements?

# In[]:


target_col = "decomposition_energy (eV/atom)"
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

new_data = data[[target_col] + [atoms_obj_col] + list(average_cols) + [separated_atoms_col]].dropna().reset_index()


# In[]:


largest_system =  new_data["atoms_object (unitless)"].apply(len).max()
sm = SineMatrix(
    n_atoms_max = largest_system,
    permutation = "eigenspectrum",
    sparse = False,
    flatten = True
)


# In[]:


def get_sm(atoms):
    new_cols = sm.create(atoms).reshape(1,-1).flatten()
    return new_cols

raw_soap = new_data["atoms_object (unitless)"].apply(get_sm)
refined_soap = np.vstack(raw_soap)
# This results on some very small (e.g. 10^-14) imaginary components. We'll remove those.
refined_soap = np.real(refined_soap)
soap_df = pd.DataFrame(refined_soap, columns=[f"sine_eigenspectrum_{i}" for i in range(sm.n_atoms_max)])
soap_df


# In[]:


data_scaled = pd.concat([new_data, soap_df], axis=1).drop(columns=["atoms_object (unitless)", "index"])
data_means = data_scaled.mean()
data_std = data_scaled.std()
data_scaled = ((data_scaled - data_means) / data_std)
data_scaled


# In[]:


data_scaled.to_csv("new_test_dataset.csv")


# In[ ]:




