#!/usr/bin/env python
# coding: utf-8

# In[]:


import itertools

import collections
import random

import sklearn.pipeline, sklearn.impute, sklearn.preprocessing
import sklearn.model_selection
import sklearn.ensemble
import sklearn.feature_selection

import numpy as np
import pandas as pd
import pymatgen.symmetry.analyzer
from pymatgen.analysis.local_env import JmolNN
import tqdm

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# In[]:


RANDOM_SEED = 1234
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tqdm.tqdm.pandas()


# In[]:


dataset_path = "../data_splitting/all_descriptors.pkl"
df = pd.read_pickle(dataset_path)


# # Spacegroup Binning

# In[]:


df["spacegroup_number"] = df["ox_struct"].apply(lambda struct: pymatgen.symmetry.analyzer.SpacegroupAnalyzer(struct).get_space_group_number())


# In[]:


counts = collections.Counter(df["spacegroup_number"])
sorted_counts = list(sorted(zip(map(str, counts.keys()), counts.values()), key=lambda i: -i[1]))
counts_x = [i[0] for i in sorted_counts]
counts_y = [i[1] for i in sorted_counts]

plt.rcParams["figure.figsize"] = [20,10]
plt.rcParams["font.size"] = 16
plt.xticks(rotation=45)

# Barplot
sns.barplot(x=counts_x, y=counts_y, palette=sns.color_palette('rocket_r', n_colors=len(counts_x)))
plt.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
plt.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
plt.gca().yaxis.grid(which='major', color="black")
plt.gca().yaxis.grid(which='minor', color="black", alpha=0.25, linestyle='--')
plt.gca().set_axisbelow(True)
plt.xlabel("Spacegroup")
plt.ylabel("Count")

# Probabilities
counts_cumulative = []
total = 0
for count in counts_y:
    total += count
    counts_cumulative.append(total)
counts_cumulative = list(map(lambda i: 100 * i / counts_cumulative[-1], counts_cumulative))

ax2 = plt.twinx()
ax2.plot(counts_cumulative, c='black', lw=3)
ax2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
ax2.set_ylim([0,100])
ax2.set_ylabel("Cumulative Count (%)")


# In[]:


counts = collections.Counter(df["spacegroup_number"])
sorted_counts = list(sorted(zip(map(str, counts.keys()), counts.values()), key=lambda i: -i[1]))

binned_counts = {"other": 0}
for spacegroup, count in sorted_counts:
    if count > 5:
        binned_counts[spacegroup] = count
    else:
        binned_counts["other"] += count

sorted_counts = list(sorted(zip(map(str, binned_counts.keys()), binned_counts.values()), key=lambda i: -i[1]))
counts_x = [i[0] for i in sorted_counts]
print(len(counts_x))
counts_y = [i[1] for i in sorted_counts]

# Barplot
sns.barplot(x=counts_x, y=counts_y, palette=sns.color_palette('rocket_r', n_colors=len(counts_x)))
plt.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
plt.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
plt.gca().yaxis.grid(which='major', color="black")
plt.gca().yaxis.grid(which='minor', color="black", alpha=0.25, linestyle='--')
plt.gca().set_axisbelow(True)
plt.xlabel("Spacegroup")
plt.ylabel("Count")

# Probabilities
counts_cumulative = []
total = 0
for count in counts_y:
    total += count
    counts_cumulative.append(total)
counts_cumulative = list(map(lambda i: 100 * i / counts_cumulative[-1], counts_cumulative))

ax2 = plt.twinx()
ax2.plot(counts_cumulative, c='black', lw=3)
ax2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
ax2.set_ylim([0,100])
ax2.set_ylabel("Cumulative Count (%)")


# In[]:


df['spacegroup_number'] = df['spacegroup_number'].apply(lambda val: binned_counts.get(str(val), 'other'))


# # Additional Structure Featurization

# In[]:


classes = {
'alkaline' : ['H', 'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr'],
'alkaine_earth' : ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra'],
'early_transition' : ['Sc', 'Ti', 'V', 'Cr', 'Mn',
                    'Y', 'Zr', 'Nb', 'Mo', 'Tc',
                    'Hf', 'Ta', 'W', 'Re', 'Os'],
'late_transition' : ['Fe', 'Co', 'Ni', 'Cu', 'Zn',
                   'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                   'Os', 'Ir', 'Pt', 'Au', 'Hg'],
'triel' : ['B', 'Al', 'Ga', 'In', 'Tl'],
'tetrel' : ['C', 'Si', 'Ge', 'Sn', 'Pb'],
'pnictogen' : ['N', 'P', 'As', 'Sb', 'Bi'],
'chalcogen' : ['O', 'S', 'Se', 'Te', 'Po'],
'halide' : ['F', 'Cl', 'Br', 'I', 'At']
}

groups = {}
for key, values in classes.items():
    for val in values:
        groups[val] = key


# In[]:


symbols_cols = collections.Counter()
bond_cols = collections.Counter()
angle_cols = collections.Counter()

neighbor_finder = JmolNN()

with tqdm.tqdm(total=len(df)) as pbar:
    for struct in df["ox_struct"]:
        symbols_cols.update([groups[symbol] for symbol in struct.symbol_set])
        
        for index, site in enumerate(struct.sites):
            connected = [i['site'] for i in neighbor_finder.get_nn_shell_info(struct, index, 1)]
            
            # Bond counts
            for vertex in connected:
                start, end = sorted([groups[str(site.specie.element)], groups[str(vertex.specie.element)]])
                bond = f"{start}-{end}"
                bond_cols[bond] += 0.5
                
            # Angles
            for angle_start, angle_end in map(sorted, itertools.combinations(connected,2)):
                angle = f"{groups[str(angle_start.specie.element)]}-{groups[str(site.specie.element)]}-{groups[str(angle_end.specie.element)]}"
                angle_cols[angle] += 1
        pbar.update(1)


# In[]:


tqdm.tqdm.pandas()
all_symbols = set(symbols_cols.keys())
all_bonds = set(bond_cols.keys())
all_angles = set(angle_cols.keys())

def featurize(data):
    symbol_units = "atoms"
    bond_units = "bonds"
    angle_units = "angles"
    struct = data["ox_struct"]
    
    present_symbols = collections.Counter([groups[symbol] for symbol in struct.symbol_set])
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
            start, end = sorted([groups[str(site.specie.element)], groups[str(vertex.specie.element)]])
            bond = f"{start}-{end}"
            present_bonds[bond] += 0.5
            
        # Count Angles
        for angle_start, angle_end in map(sorted, itertools.combinations(connected, 2)):
            angle = f"{groups[str(angle_start.specie.element)]}-{groups[str(site.specie.element)]}-{groups[str(angle_end.specie.element)]}"
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

all_data_features = df.progress_apply(featurize, axis=1)
all_data_features


# # Feature Selection

# In[]:


object_cols = ["atoms_object (unitless)",
               "ox_struct"]

regression_irrelevant = object_cols + [
    'discovery_process (unitless)',
    'potcars (unitless)',
    'is_hubbard (unitless)',
    'energy_per_atom (eV)',
    'decomposition_energy (eV/atom)',
    'is_bandgap_direct (unitless)',
    'is_metal (unitless)',
    'energy_vdw_per_atom (eV/atom)',
    'total_magnetization (Bohr Magneton)']

train, test = sklearn.model_selection.train_test_split(all_data_features.drop(columns = regression_irrelevant), test_size=0.1, random_state=RANDOM_SEED)


# In[]:


data_train, data_test = sklearn.model_selection.train_test_split(train.drop(columns=['formula', '2dm_id (unitless)', 'exfoliation_energy_per_atom (eV/atom)']).fillna(0), 
                                                                 test_size=0.1, random_state=RANDOM_SEED)

train_x = data_train.drop(columns=["bandgap (eV)", 'spacegroup_number']).to_numpy()
train_y = data_train["bandgap (eV)"].to_numpy()

val_x = data_test.drop(columns=["bandgap (eV)", 'spacegroup_number']).to_numpy()
val_y = data_test["bandgap (eV)"].to_numpy()

model = sklearn.pipeline.Pipeline(
    [
     ("Scaler", sklearn.preprocessing.MinMaxScaler()),
     ("Composites", sklearn.preprocessing.PolynomialFeatures(degree=2, interaction_only=True)),
     ("Ensemble", sklearn.ensemble.RandomForestRegressor(n_estimators = 100, max_features=50)
     )]
)


# In[]:


model.fit(X=train_x, y=train_y)

train_y_pred = model.predict(train_x)
val_y_pred = model.predict(val_x)

# Plot the results   
plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
plt.rcParams["figure.figsize"] = (10,10)
plt.rcParams["font.size"] = 16

plt.scatter(x=train_y_pred, y=train_y, label="Train Set")
plt.scatter(x=val_y_pred, y=val_y, label="Validation Set")
min_xy = min(min(train_y_pred), min(train_y))
max_xy = max(max(train_y_pred), max(train_y))

plt.plot([min_xy,max_xy], [min_xy,max_xy], label="Parity")
plt.ylabel("Bandgap (Dataset)")
plt.xlabel("Bandgap (Predicted)")
plt.legend()
plt.show()


# In[]:


importances = list(sorted(zip(data_train.drop(columns="bandgap (eV)").columns, model[1].feature_importances_), key=lambda i: -i[1]))
importances


# In[]:


n_features = 25
importances[:n_features]


# In[]:


mutual_information = list(sorted(zip(data_train.drop(columns="bandgap (eV)").columns, sklearn.feature_selection.mutual_info_regression(train_x, train_y)), key=lambda i: -i[1]))
mutual_information


# In[]:


mutual_information[:n_features]


# In[]:


importants = set([i[0] for i in importances[:n_features]])
mutual_infos = set([i[0] for i in mutual_information[:n_features]])

set.intersection(importants, mutual_infos)


# In[]:


exported_features = [i[0] for i in importances[:n_features]]
print(exported_features)


# In[]:


exported_features_units = {
    "ave:density": "mass/length^3",
    "max:heat_capacity_mass": "energy/mass*temperature",
    "min:atomic_weight": "mass",
    "bond_length_average": "length",
    "min:num_valance": "count",
    "ave:atomic_number": "count",
    "max:density": "mass/length^3",
    "ave:boiling_point": "temperature",
    "ave:gs_energy": "energy",
    "ave:bulk_modulus": "force/length^2",
    "max:melting_point": "temperature",
    "var:vdw_radius_alvarez": "length",
    "ave:atomic_weight": "mass",
    "var:atomic_radius": "length",
    "min:atomic_number": "count",
    "max:bulk_modulus": "force/length^2",
    "ave:melting_point": "temperature",
    "var:boiling_point": "temperature",
    "ave:heat_of_formation": "energy/count",
    "ave:num_d_valence": "count",
    "var:gs_bandgap": "energy",
    "var:en_ghosh": "energy",
    "min:Polarizability": "length^3",
    "global_instability": "unitless",
    "ave:vdw_radius": "length",
    "spacegroup_number": "unitless"
}


# In[]:


def add_units(colname):
    if colname not in ["bandgap (eV)"]:
        return f"{colname} ({exported_features_units[colname]})"
    else:
        return colname


# In[]:


train_export = train[['bandgap (eV)', 'spacegroup_number'] + exported_features].fillna(0)
train_export = train_export.rename(add_units, axis=1)
train_export.to_csv('data_train_featurized_importances_bandgap.csv')


# In[]:


test_export = test[['bandgap (eV)', 'spacegroup_number'] + exported_features].fillna(0)
test_export = test_export.rename(add_units, axis=1)
test_export.to_csv('data_test_featurized_importances_bandgap.csv')


# # Testing SISSO

# In[ ]:


class model:
    def __init__(fun, coefs_dict):
        

