#!/usr/bin/env python
# coding: utf-8

# In[]:


import random

import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection
import rdkit.Chem
from rdkit.Chem import Descriptors

# Drop console spam when molecules are featurized
rdkit.RDLogger.DisableLog("rdApp.*")


# In[]:


dataset_path = "raw_data/bitter-sweet.csv"
raw_data = pd.read_csv(dataset_path)


# In[]:


data = raw_data[["Name", "Canonical SMILES"]].copy()

# Let's take a 1 v all approach to this, and just encode "Bitter" as 1 and everything else as 0.
classes = ("Sweet", "Bitter", "Tasteless", "Non-bitter")
binarized = sklearn.preprocessing.label_binarize(raw_data.Taste.to_numpy(), classes=classes)
bitter = binarized[:,classes.index("Bitter")]

data["Bitter"] = bitter
data = data.dropna()
for colname in data.drop(columns=["Bitter"]).columns:
    data = data.dropna().drop_duplicates(subset=colname)

data.head()


# In[]:


data["Molecule Objects"]=data["Canonical SMILES"].apply(rdkit.Chem.MolFromSmiles)
data = data.dropna().copy()
data.head()

cls_data = data[["Bitter"]].copy()
cls_data.head()


# In[]:


# Ignore the fragment features for now
features = filter(lambda i: not i[0].startswith("fr_"), Descriptors.descList)
for name, fun in features:
    # If we can't apply a feature, just ignore it
    try:
        cls_data[name] = data["Molecule Objects"].apply(fun)
    except ZeroDivisionError:
        pass
cls_data.head()


# In[]:


def drop_low_unique_columns(dataset, cutoff=2, target_col = "Bitter"):
    mask = dataset.nunique() > cutoff
    mask.Bitter=True
    return dataset[dataset.columns[mask]]

cls_data = drop_low_unique_columns(cls_data).reset_index().drop(columns=["index"])


# In[]:


# Also drop stuff with an "infinite" (overflowed) standard deviation
cls_data = cls_data.drop(columns=cls_data.columns[cls_data.std()==np.inf])


# In[]:


random.seed(1234)
np.random.seed(1234)

data_train, data_test = sklearn.model_selection.train_test_split(cls_data, test_size=0.2)

data_train_x = data_train.drop(columns=["Bitter"])
data_train_y = data_train.Bitter

mean = data_train_x.mean()
std = data_train_x.std()

data_test_x = data_test.drop(columns=["Bitter"])
data_test_y = data_test.Bitter


# In[]:


data_train_scaled = pd.concat([data_train_y, (data_train_x - mean) / std], axis=1)
data_test_scaled = pd.concat([data_test_y, (data_test_x - mean) / std], axis=1)


# In[]:


mean.to_pickle("dataset_means_stds/bittersweet_mean.pkl")
std.to_pickle("dataset_means_stds/bittersweet_std.pkl")

data_train_scaled.to_csv("scaled_featurized_train/scaled_bittersweet_train.csv")
data_test_scaled.to_csv("scaled_featurized_test/scaled_bittersweet_test.csv")


# In[ ]:




