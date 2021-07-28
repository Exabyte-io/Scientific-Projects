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
import sklearn.cluster
import sklearn.decomposition

import numpy as np
import pandas as pd
import pymatgen.symmetry.analyzer
from pymatgen.analysis.local_env import JmolNN
import tqdm

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import ase # Needs to be < 3.19.3


# In[]:


RANDOM_SEED = 1234
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tqdm.tqdm.pandas()


# In[]:


dataset_path = "../data_splitting/all_descriptors.pkl"
df = pd.read_pickle(dataset_path)


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


# # Train/Test Split and Clustering

# In[]:


to_fill = [col for col in all_data_features.columns if any([col.endswith('(atoms)'), col.endswith('(bonds)'), col.endswith('(angles)')])]
all_data_features[to_fill] = all_data_features[to_fill].fillna(0)

train, test = sklearn.model_selection.train_test_split(all_data_features, test_size=0.1, random_state=RANDOM_SEED)


# In[]:


import dscribe.descriptors

symbols = range(1,93)

soap = dscribe.descriptors.SOAP(species=symbols,
                                periodic=True,
                                rcut=4,
                                nmax=2,
                                lmax=4)
def saponify(atoms):
    lathered = soap.create(atoms)
    # Soap creates an N x M array
    #     - N is the number of atoms in the system
    #     - M is the size of the SOAP descriptor 
    # So we'll average along the N direction
    rinsed = np.hstack([lathered.mean(axis=0), lathered.min(axis=0), lathered.max(axis=0)])
    return rinsed


# In[]:


cluster_training_data = np.vstack(train['atoms_object (unitless)'].progress_apply(saponify).to_numpy())
print(cluster_training_data.shape)

pca = sklearn.decomposition.PCA(n_components=256)
cluster_training_data = pca.fit_transform(cluster_training_data)
print(cluster_training_data.shape)


# In[]:


plt.plot(cluster_training_data.mean(axis=0), c='k', label="Mean")
plt.fill_between(range(cluster_training_data.shape[1]), cluster_training_data.min(axis=0), cluster_training_data.max(axis=0), color='#FF5555', label="Extrema")
plt.xlabel("Principle Component")
plt.ylabel("Value (Arbitrary)")
plt.legend()


# In[]:


results = {}
min_clusters=2
max_clusters=11
print("N_Clusters\tError")
for n_clusters in range(min_clusters, max_clusters):
    clusters = sklearn.cluster.KMeans(n_clusters=n_clusters,
                                      n_init=16,
                                      random_state=RANDOM_SEED)
    clusters.fit(cluster_training_data)
    error = sklearn.metrics.calinski_harabasz_score(cluster_training_data, clusters.predict(cluster_training_data))
    results[n_clusters] = error
    print(n_clusters, error, sep="\t\t")
    
plt.plot(list(results.keys()), list(results.values()), marker="o", color="black")
plt.xticks(range(min_clusters, max_clusters))
plt.xlabel("N Clusters")
plt.ylabel("Calinski Harabasz Score")
plt.show()


# In[]:


n_clusters = max(results, key=results.get)
print(n_clusters)

clusters = sklearn.cluster.KMeans(n_clusters=n_clusters,
                                  n_init=8,
                                  random_state=RANDOM_SEED)
clusters.fit(cluster_training_data)

labels = collections.Counter(clusters.predict(cluster_training_data))

# We can't actually run SISSO on the dataset if we don't have enough examples for all task labels
while any([count < 4 for count in labels.values()]):
    n_clusters -= 1
    clusters = sklearn.cluster.KMeans(n_clusters=n_clusters,
                                      n_init=8,
                                      random_state=42)
    clusters.fit(cluster_training_data)
    labels = collections.Counter(clusters.predict(cluster_training_data))
print(n_clusters)


# In[]:


cluster_test = np.vstack(test['atoms_object (unitless)'].progress_apply(saponify).to_numpy())
train['soap_label'] = clusters.predict(cluster_training_data)
test['soap_label'] = clusters.predict(pca.transform(cluster_test))


# In[]:


collections.Counter(train['soap_label'])


# In[]:


collections.Counter(test['soap_label'])


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


# In[]:


data_train, data_test = sklearn.model_selection.train_test_split(train.drop(columns=regression_irrelevant + ['formula', '2dm_id (unitless)', 'exfoliation_energy_per_atom (eV/atom)']).fillna(0), 
                                                                 test_size=0.1, random_state=RANDOM_SEED)

train_x = data_train.drop(columns=["bandgap (eV)", 'soap_label']).to_numpy()
train_y = data_train["bandgap (eV)"].to_numpy()

val_x = data_test.drop(columns=["bandgap (eV)", 'soap_label']).to_numpy()
val_y = data_test["bandgap (eV)"].to_numpy()

model = sklearn.pipeline.Pipeline(
    [("Scaler", sklearn.preprocessing.MinMaxScaler()),
     ("Ensemble", sklearn.ensemble.RandomForestRegressor(n_estimators = 100, max_features=50))
    ]
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


train_export = train[['bandgap (eV)', 'soap_label'] + exported_features].fillna(0).sort_values('soap_label')
train_export.to_csv('data_train_featurized_soap_importances_bandgap.csv')


# In[]:


test_export = test[['bandgap (eV)', 'soap_label'] + exported_features].fillna(0).sort_values('soap_label')
test_export.to_csv('data_test_featurized_soap_importances_bandgap.csv')


# In[ ]:




