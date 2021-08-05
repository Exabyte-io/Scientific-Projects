#!/usr/bin/env python
# coding: utf-8

# In[]:


import random

import numpy as np
import pandas as pd
import sklearn.model_selection


# In[]:


dataset_path = "raw_data/curated-solubility-dataset.csv"
raw_data = pd.read_csv(dataset_path)
raw_data.head()


# In[]:


cols_to_drop=["ID", "Name", "InChI", "InChIKey", "SD", "Ocurrences", "Group", "SMILES"]
data = raw_data.drop(columns=cols_to_drop).dropna().reset_index().drop(columns="index")
data.head()


# In[]:


random.seed(1234)
np.random.seed(1234)

data_train, data_test = sklearn.model_selection.train_test_split(data, test_size=0.2)


# In[]:


mean = data_train.mean()
std = data_train.std()

data_train_scaled = (data_train - mean) / std
data_test_scaled = (data_test - mean) / std


# In[]:


mean.to_pickle("dataset_means_stds/logS_mean.pkl")
std.to_pickle("dataset_means_stds/logS_std.pkl")

data_train_scaled.to_csv("scaled_featurized_train/scaled_logS_train.csv")
data_test_scaled.to_csv("scaled_featurized_test/scaled_logS_test.csv")


# In[]:


foo = data_train_scaled[data_train_scaled["MolWt"]<=1]


# In[]:


foo[foo["MolWt"]>=-1]


# In[]:


data_train_scaled.describe()


# In[ ]:


data

