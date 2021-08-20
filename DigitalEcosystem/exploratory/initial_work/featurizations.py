#!/usr/bin/env python
# coding: utf-8

# In[]:


import pandas as pd


# In[]:


import functools
import random

import pandas as pd
import numpy as np

import sklearn.preprocessing
import sklearn.feature_selection
import sklearn.pipeline

import ase
import ase.neighborlist

from dscribe.descriptors import SineMatrix


# In[]:


# Read the data
datafile = "raw_data/2d_mat_dataset_raw.pkl"
data = pd.read_pickle(datafile)
initial_size = len(data)
data.head()
featurized = data[["atoms_object (unitless)", "bandgap (eV)"]]
featurized


# In[]:


x = featurized["atoms_object (unitless)"][0]


# In[]:


all_atoms = ase.Atoms()
for atom in 


# In[]:


atoms += x


# In[ ]:




