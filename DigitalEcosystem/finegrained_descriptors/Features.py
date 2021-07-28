#!/usr/bin/env python
# coding: utf-8

# In[]:


import itertools
import collections
import random
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tqdm
import pymatgen.core
import pymatgen.io.ase
import pymatgen.analysis.dimensionality
from pymatgen.analysis.local_env import JmolNN


RANDOM_SEED = 1234

pd.options.mode.chained_assignment = None
tqdm.tqdm.pandas()
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# In[]:


raw_df = pd.read_pickle("../DigitalEcosystem/raw_data/2d_mat_dataset_raw.pkl")
cols_to_keep = ["bandgap (eV)", "atoms_object (unitless)"]
df = raw_df[cols_to_keep]


# In[]:


df["pymatgen_structure (unitless)"] = df["atoms_object (unitless)"].apply(pymatgen.io.ase.AseAtomsAdaptor.get_structure)


# In[]:


symbols_cols = collections.Counter()
bond_cols = collections.Counter()
angle_cols = collections.Counter()

neighbor_finder = JmolNN()

with tqdm.tqdm(total=len(df)) as pbar:
    for struct in df["pymatgen_structure (unitless)"]:
        symbols_cols.update(struct.symbol_set)
        
        for index, site in enumerate(struct.sites):
            connected = [i['site'] for i in neighbor_finder.get_nn_shell_info(struct, index, 1)]
            
            # Bond counts
            for vertex in connected:
                start, end = sorted([site.specie, vertex.specie])
                bond = f"{start}-{end}"
                bond_cols[bond] += 0.5
                
            # Angles
            for angle_start, angle_end in map(sorted, itertools.combinations(connected,2)):
                angle = f"{angle_start.specie}-{site.specie}-{angle_end.specie}"
                angle_cols[angle] += 1
        pbar.update(1)


# In[]:


for filename, obj in (("symbols.pkl", symbols_cols),
                      ("bonds.pkl", bond_cols),
                      ("angles.pkl", angle_cols)):
    with open(filename, "wb") as outp:
        pickle.dump(obj, outp)


# In[]:


def testmap(foo):
    foo[random.choice(["A", "B"])] = 1
    return foo
test = df.head().copy()
test.apply(testmap, axis=1)


# In[]:


all_symbols = set(symbols_cols.keys())
all_bonds = set(bond_cols.keys())
all_angles = set(angle_cols.keys())

def featurize(data):
    symbol_units = "atoms"
    bond_units = "bonds"
    angle_units = "angles"
    struct = data["pymatgen_structure (unitless)"]
    
    present_symbols = collections.Counter(struct.symbol_set)
    present_bonds = collections.Counter()
    present_angles = collections.Counter()
    
    # Record and Count Symbols
    for symbol, count in present_symbols.items():
        data[f"{symbol} ({symbol_units})"] = count
    data[f"Total Atoms ({symbol_units})"] = sum(present_symbols.values())
    
    for index, site in enumerate(struct.sites):
        connected = [i['site'] for i in neighbor_finder.get_nn_shell_info(struct, index, 1)]
        
        # Count Bonds
        for vertex in connected:
            start, end = sorted([site.specie, vertex.specie])
            bond = f"{start}-{end}"
            present_bonds[bond] += 0.5
            
        # Count Angles
        for angle_start, angle_end in map(sorted, itertools.combinations(connected, 2)):
            angle = f"{angle_start.specie}-{site.specie}-{angle_end.specie}"
            present_angles[angle] += 1
            
    # Record Bonds
    for bond, count in present_bonds.items():
        data[f"{bond} ({bond_units})"] = count
    data[f"Total Bonds ({bond_units})"] = sum(present_bonds.values())
            
    # Record Angles
    for angle, count in present_angles.items():
        data[f"{angle} ({angle_units})"] = count
    data[f"Total Angles ({angle_units})"] = sum(present_angles.values())
    
    return data

all_data_features = df.progress_apply(featurize, axis=1).fillna(0)
all_data_features


# In[]:


all_data_features.to_pickle("all_data_features.pkl")


# In[ ]:




