#!/usr/bin/env python
# coding: utf-8

# In[]:


import collections
import random

import numpy as np
import pandas as pd

import sklearn.model_selection
from xenonpy.descriptor import Compositions


# In[]:


# Read data
data_path = "raw_data/perovskites.pkl"
data = pd.read_pickle(data_path)
data.head()


# In[]:


data["Volume"] /= data["Atoms_Object"].apply(lambda atoms: len(atoms)//5)

# Featurize with XenonPy
cal = Compositions(featurizers=["WeightedAverage"])
data["Symbols"] = data.Atoms_Object.apply(lambda atoms: collections.Counter(atoms.get_chemical_symbols()))
featurized_data = pd.concat([data, cal.transform(data.Symbols)], axis=1)


# In[]:


data = featurized_data.drop(columns=["Formula", "Atoms_Object", "Symbols"])

# Train/Test Split
np.random.seed(1234)
random.seed(1234)

train, test = sklearn.model_selection.train_test_split(data, test_size=0.2)
mean = train.mean()
std = train.std()


# In[]:


train_scaled = ((train - mean) / std).dropna(axis=1)
test_scaled = ((test - mean) / std).dropna(axis=1)


# In[]:


mean.to_pickle("dataset_means_stds/perov_mean.pkl")
std.to_pickle("dataset_means_stds/perov_std.pkl")

train_scaled.to_csv("scaled_featurized_train/scaled_perovskite_train.csv")
test_scaled.to_csv("scaled_featurized_test/scaled_perovskite_test.csv")


# In[ ]:




