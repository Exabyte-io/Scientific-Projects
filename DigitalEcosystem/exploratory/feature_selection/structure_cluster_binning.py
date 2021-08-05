#!/usr/bin/env python
# coding: utf-8

# In[]:


import itertools
from functools import partial

import collections
import random
import warnings

import numpy as np
import pandas as pd
import pymatgen.symmetry.analyzer

from pymatgen.analysis.local_env import JmolNN
import tqdm

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import ase # Needs to be <= 3.19.3
import swifter

import pymatgen.core.composition
import pymatgen.io.ase
from pymatgen.analysis.local_env import JmolNN

import sklearn.pipeline, sklearn.impute, sklearn.preprocessing
import sklearn.model_selection
import sklearn.ensemble
import sklearn.feature_selection
import sklearn.cluster
import sklearn.decomposition

import matminer
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.structure.misc import EwaldEnergy
from matminer.featurizers.structure.order import StructuralComplexity
from matminer.featurizers.structure.symmetry import GlobalSymmetryFeatures
from matminer.featurizers.structure.bonding import GlobalInstabilityIndex
from matminer.featurizers.site.bonding import AverageBondLength


# In[]:


RANDOM_SEED = 1234
pd.options.mode.chained_assignment = None
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tqdm.tqdm.pandas()


# # Read and Clean the Data

# In[]:


dataset_path = "../../raw_data/2d_mat_dataset_raw.pkl"
df = pd.read_pickle(dataset_path)

total = len(df)
print(f"Starting with {total} entries. Includes top-down and bottom-up.")

# Remove systems containing f-block / synthetic elements
fblock_and_synth = ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
                    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
                    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Tg", "Cn", "Nh", "Fl", "Mc", "Mc", "Lv", "Ts", "Og"]
df = df[df["atoms_object (unitless)"].apply(
           lambda atoms: all(symbol not in fblock_and_synth for symbol in atoms.get_chemical_symbols()))
       ]
print(f"Discarding {total - len(df)} entries in the f-block, or synthetics. Total is now {len(df)}.")
total = len(df)

# Remove systems containing f-block / synthetic elements
nobles = ["He", "Ne", "Ar", "Kr", "Xe", "Rn"]
df = df[df["atoms_object (unitless)"].apply(
           lambda atoms: all(symbol not in nobles for symbol in atoms.get_chemical_symbols()))
       ]
print(f"Discarding {total - len(df)} entries with noble gases. Total is now {len(df)}.")
total = len(df)

# To keep the methodology consistent, discard systems where a U-correction was applied
df = df[df["is_hubbard (unitless)"] == False]
print(f"Discarding {total - len(df)} entries that have U-corrections applied (metal oxides/fluorides). Total is now {len(df)}.")
total = len(df)

# Also like, keep only S
df = df[df["atoms_object (unitless)"].apply(lambda atoms: "S" in atoms.get_chemical_symbols())]
print(f"Discarding {total - len(df)} entries that don't contain Sulfur. Total is now {len(df)}.")
total = len(df)


# # Featurization

# In[]:


structures = df['atoms_object (unitless)'].apply(pymatgen.io.ase.AseAtomsAdaptor.get_structure)
df['ox_struct'] = structures.apply(lambda i: i.copy())

# struct.add_oxidation_state_by_guess() modifies the structure in-place
df.ox_struct.progress_apply(lambda struct: struct.add_oxidation_state_by_guess())


# In[]:


struct_features = MultipleFeaturizer([
    EwaldEnergy(),
    StructuralComplexity(),
    GlobalSymmetryFeatures('n_symmetry_ops'),
])
df[struct_features.feature_labels()] = struct_features.featurize_many(df.ox_struct).copy()


# In[]:


desc = GlobalInstabilityIndex()

def maybe_global_instability(struct):
    try:
        return desc.featurize(struct)[0]
    except:
        return None

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    df['global_instability'] = df.ox_struct.progress_apply(maybe_global_instability).copy()


# In[]:


neighbor_finder = JmolNN()

def average_bond_length(structure, featurizer = AverageBondLength(neighbor_finder)):
    n_atoms = len(structure)
    try:
        lengths = map(lambda i: featurizer.featurize(structure, i)[0], range(n_atoms))
        return sum(lengths) / n_atoms
    except IndexError:
        return None

df['bond_length_average'] = df.ox_struct.progress_apply(average_bond_length).copy()


# In[]:


from matminer.featurizers.site.bonding import AverageBondAngle

def average_bond_angle(structure, featurizer = AverageBondAngle(neighbor_finder)):
    n_atoms = len(structure)
    try:
        angles = map(lambda i: featurizer.featurize(structure, i)[0], range(n_atoms))
        return sum(angles) / n_atoms
    except IndexError:
        return None
df['bond_angle_average'] = df.ox_struct.progress_apply(average_bond_angle).copy()


# In[]:


def average_cn(structure, neighbor_finder = neighbor_finder):
    n_atoms = len(structure)
    cns = map(lambda i: neighbor_finder.get_cn(structure, i), range(n_atoms))
    return sum(cns) / n_atoms
df['average_cn'] = df.ox_struct.progress_apply(average_cn).copy()


# In[]:


def ab_perimeter_area_ratio(structure):
    a, b, c = structure.lattice.matrix
    perimeter = 2*np.linalg.norm(a) + 2*np.linalg.norm(b)
    area = np.linalg.norm(np.cross(a,b))
    return perimeter / area
df['perimeter_area_ratio'] = df.ox_struct.progress_apply(ab_perimeter_area_ratio).copy()


# In[]:


df["formula"] = df["atoms_object (unitless)"].progress_apply(lambda atoms: atoms.get_chemical_formula(empirical=True))


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

pca = sklearn.decomposition.PCA(n_components=128)
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
min_clusters = 2
max_clusters = 32
n_skip = 1
print("N_Clusters\tError")
for n_clusters in np.arange(min_clusters, max_clusters, n_skip):
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


data_train, data_test = sklearn.model_selection.train_test_split(train[train['soap_label']==1].drop(columns=regression_irrelevant + ['formula', '2dm_id (unitless)', 'exfoliation_energy_per_atom (eV/atom)']).fillna(0),
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

plt.title("Type 1")
plt.plot([min_xy,max_xy], [min_xy,max_xy], label="Parity")
plt.ylabel("Bandgap (Dataset)")
plt.xlabel("Bandgap (Predicted)")
plt.legend()
plt.show()


# In[]:


data_train, data_test = sklearn.model_selection.train_test_split(train[train['soap_label']==0].drop(columns=regression_irrelevant + ['formula', '2dm_id (unitless)', 'exfoliation_energy_per_atom (eV/atom)']).fillna(0),
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

plt.title("Type 0")
plt.plot([min_xy,max_xy], [min_xy,max_xy], label="Parity")
plt.ylabel("Bandgap (Dataset)")
plt.xlabel("Bandgap (Predicted)")
plt.legend()
plt.show()


# In[]:


importances = list(sorted(zip(data_train.drop(columns="bandgap (eV)").columns, model[1].feature_importances_), key=lambda i: -i[1]))
mutual_information = list(sorted(zip(data_train.drop(columns="bandgap (eV)").columns, sklearn.feature_selection.mutual_info_regression(train_x, train_y)), key=lambda i: -i[1]))


# In[]:


n_features = 25

importants = set([i[0] for i in importances[:n_features]])
mutual_infos = set([i[0] for i in mutual_information[:n_features]])

set.intersection(importants, mutual_infos)


# In[]:


exported_features = [i[0] for i in importances[:n_features]]
print(exported_features)


# In[]:


train_export = train[['bandgap (eV)', 'soap_label'] + exported_features].fillna(0).sort_values('soap_label')
#train_export.to_csv('data_train_featurized_sulfurUnrestricted_soap_importances_bandgap.csv')


# In[]:


test_export = test[['bandgap (eV)', 'soap_label'] + exported_features].fillna(0).sort_values('soap_label')
#test_export.to_csv('data_test_featurized_sulfurUnrestricted_soap_importances_bandgap.csv')


# # SISSO Models

# In[]:


#R1D1
r1d1_t0 = lambda df: 1.850631418740413e+00 +            1.560888804033170e-11 * (df['var:sound_velocity'] * df['sum:hhi_r'])
r1d1_t1 = lambda df: 7.593072978523676e-01 + 8.053457188249878e-13 * (df['var:sound_velocity'] * df['sum:hhi_r'])

#R1D2
r1d2_t0 = lambda df: 2.846720668897464e+00 +            -1.002110960463248e+00 * (df['average_cn'] / df['ave:period']) +            1.476693682392120e-11 * (df['var:sound_velocity'] * df['sum:hhi_r'])
r1d2_t1 = lambda df: 1.755560174422515e+00 + -9.219908571141404e-01 * (df['average_cn'] / df['ave:period']) +            7.687571882326259e-13 * (df['var:sound_velocity'] * df['sum:hhi_r'])

# R1D3
r1d3_t0 = lambda df: 3.336572064073060e+00 +            7.765011870062933e-05 * (df['sum:heat_of_formation'] * df['average_cn']) +            -2.258678345200038e+00 * (df['average_cn'] / df['ave:period']) +            1.181395466097512e-11 * (df['var:sound_velocity'] * df['sum:hhi_r'])
r1d3_t1 = lambda df: 1.779675579057679e+00 +            -1.293263119117848e-05 * (df['sum:heat_of_formation'] * df['average_cn']) +            -7.968776670442069e-01 * (df['average_cn'] / df['ave:period']) +            9.117578085106980e-13 * (df['var:sound_velocity'] * df['sum:hhi_r'])

# R1D4
r1d4_t0 = lambda df:4.827220821512196e+00 +           -1.915451480137068e-01 * abs(df['sum:en_pauling'] - df['sum:gs_est_bcc_latcnt']) +           1.184567860422892e-04 * (df['sum:heat_of_formation'] * df['average_cn']) +           -3.594619637535246e+00 * (df['average_cn'] / df['ave:period']) +           1.174504091334793e-11 * (df['var:sound_velocity'] * df['sum:hhi_r'])
r1d4_t1 = lambda df: 1.600949474929033e+00 +           8.833146157869971e-03 * abs(df['sum:en_pauling'] - df['sum:gs_est_bcc_latcnt']) +            -1.985610749739964e-05 * (df['sum:heat_of_formation'] * df['average_cn']) +           -6.663877215670986e-01 * (df['average_cn'] / df['ave:period']) +            9.393695287462473e-13 * (df['var:sound_velocity'] * df['sum:hhi_r'])

# R2D1
r1d1_t0 = lambda df: -9.348742838125810e-01 +           3.656079572576012e+00 * ((df['sum:num_p_unfilled'] / df['sum:gs_est_bcc_latcnt'])+(df['ave:specific_heat'] * df['var:electron_negativity']))
r1d1_t1 = lambda df: -1.780585286402868e-01 +           1.473338756072956e+00 * ((df['sum:num_p_unfilled'] / df['sum:gs_est_bcc_latcnt'])+(df['ave:specific_heat'] * df['var:electron_negativity']))

# R2D2
r2d2_t0 = lambda df: -3.237441842433513e-01 +           -2.014921555491347e+00 + abs((df['ave:period'] / df['average_cn']) - (df['sum:heat_of_formation'] / df['sum:melting_point'])) +           3.940275314156243e+00 * ((df['sum:num_p_unfilled'] / df['sum:gs_est_bcc_latcnt']) + (df['ave:specific_heat'] * df['var:electron_negativity']))
r2d2_t1 = lambda df: -3.479715321513178e-01 +           4.283831194623857e-01 + abs((df['ave:period'] / df['average_cn']) - (df['sum:heat_of_formation'] / df['sum:melting_point'])) +           1.313019627531759e+00 * ((df['sum:num_p_unfilled'] / df['sum:gs_est_bcc_latcnt']) + (df['ave:specific_heat'] * df['var:electron_negativity']))

# R2D3
r2d3_t0 = lambda df: 4.488510497104689e-01 +           -1.843752123832672e+00 + abs((df['ave:period'] / df['average_cn']) - (df['sum:heat_of_formation'] / df['sum:melting_point'])) +           2.846878471611191e-05 * (np.sqrt(df['var:sound_velocity']) * (df['var:electron_negativity'] * df['sum:specific_heat'])) +          2.723558937301273e+00 * ((df['sum:num_p_unfilled'] / df['sum:gs_est_bcc_latcnt']) + (df['ave:specific_heat'] * df['var:electron_negativity']))
r2d3_t1 = lambda df: -2.091580173350305e-01 +           3.215177162451530e-01 + abs((df['ave:period'] / df['average_cn']) - (df['sum:heat_of_formation'] / df['sum:melting_point'])) +           4.494951906226667e-05 * (np.sqrt(df['var:sound_velocity']) * (df['var:electron_negativity'] * df['sum:specific_heat'])) +          1.116715961404604e+00 * ((df['sum:num_p_unfilled'] / df['sum:gs_est_bcc_latcnt']) + (df['ave:specific_heat'] * df['var:electron_negativity']))


# In[]:


def fun_branch(fun0, fun1, df):
    if df['soap_label'] == 0:
        return fun0(df)
    elif df['soap_label'] == 1:
        return fun1(df)

train_export['r1d1'] = train_export.apply(partial(fun_branch, r1d1_t0, r1d1_t1), axis=1)
train_export['r1d2'] = train_export.apply(partial(fun_branch, r1d2_t0, r1d2_t1), axis=1)
train_export['r1d3'] = train_export.apply(partial(fun_branch, r1d3_t0, r1d3_t1), axis=1)
train_export['r1d4'] = train_export.apply(partial(fun_branch, r1d4_t0, r1d4_t1), axis=1)

train_export['r2d1'] = train_export.apply(partial(fun_branch, r2d1_t0, r2d1_t1), axis=1)
train_export['r2d2'] = train_export.apply(partial(fun_branch, r2d2_t0, r2d2_t1), axis=1)
train_export['r2d3'] = train_export.apply(partial(fun_branch, r2d3_t0, r2d3_t1), axis=1)

test_export['r1d1'] = test_export.apply(partial(fun_branch, r1d1_t0, r1d1_t1), axis=1)
test_export['r1d2'] = test_export.apply(partial(fun_branch, r1d2_t0, r1d2_t1), axis=1)
test_export['r1d3'] = test_export.apply(partial(fun_branch, r1d3_t0, r1d3_t1), axis=1)
test_export['r1d4'] = test_export.apply(partial(fun_branch, r1d4_t0, r1d4_t1), axis=1)

test_export['r2d1'] = test_export.apply(partial(fun_branch, r2d1_t0, r2d1_t1), axis=1)
test_export['r2d2'] = test_export.apply(partial(fun_branch, r2d2_t0, r2d2_t1), axis=1)
test_export['r2d3'] = test_export.apply(partial(fun_branch, r2d3_t0, r2d3_t1), axis=1)


# In[]:


train_export = train_export.replace([np.inf, -np.inf], np.nan)
test_export = test_export.replace([np.inf, -np.inf], np.nan)


# In[]:


plt.rcParams['figure.figsize'] = [10,10]
plt.rcParams['font.size'] = 16


# In[]:


def make_plot(colname):
    train_t0 = train_export[train_export['soap_label'] == 0]
    train_t1 = train_export[train_export['soap_label'] == 1]
    test_t0 = test_export[test_export['soap_label'] == 0]
    test_t1 = test_export[test_export['soap_label'] == 1]

    plt.scatter(x=train_t0[colname], y=train_t0['bandgap (eV)'], label="Train_Type0", marker='o', c='#FFAAAA')
    plt.scatter(x=train_t1[colname], y=train_t1['bandgap (eV)'], label="Train_Type1", marker='o', c='#AAAAFF')

    plt.scatter(x=test_t0[colname], y=test_t0['bandgap (eV)'], label="Test_Type0", marker='o',    c='#FF0000')
    plt.scatter(x=test_t1[colname], y=test_t1['bandgap (eV)'], label="Test_Type1", marker='o',    c='#0000FF')
    lims = [min(min(train_export[colname]), min(train_export['bandgap (eV)'])),
            max(max(train_export[colname]), max(train_export['bandgap (eV)']))]
    plt.plot([lims[0],lims[1]], [lims[0],lims[1]], label="Parity", ls='--', c='k')

    plt.title(colname)
    plt.ylabel("Bandgap (Actual)")
    plt.xlabel("Bandgap (Predicted)")
    plt.legend()
    plt.show()


# In[]:


make_plot('r1d1')


# In[]:


make_plot('r1d2')


# In[]:


make_plot('r1d3')


# In[]:


make_plot('r1d4')


# In[]:


make_plot('r2d1')


# In[]:


make_plot('r2d2')


# In[]:


make_plot('r2d3')


# # Histplots

# In[]:


import seaborn as sns


# In[]:


nbin = 20
sns.histplot(train[train['soap_label']==0]['bandgap (eV)'], bins=nbin, color='#FF0000', label='Test_Type0', stat='probability')
sns.histplot(train[train['soap_label']==1]['bandgap (eV)'], bins=nbin, color='#00FF00', label='Test_Type1', stat='probability')
sns.histplot(train[train['soap_label']==2]['bandgap (eV)'], bins=nbin, color='#0000FF', label='Test_Type2', stat='probability')
sns.histplot(train[train['soap_label']==3]['bandgap (eV)'], bins=nbin, color='#AAAAAA', label='Test_Type3', stat='probability')


plt.legend()


# In[]:


train[train['soap_label']==0][['2dm_id (unitless)', 'bandgap (eV)']]


# In[]:


test[test['soap_label']==1]


# In[]:


import tpot


# In[]:


model = tpot.TPOTRegressor(
    generations=None,
    population_size=100,
    max_eval_time_mins=10/60,
    max_time_mins=1,
    cv=5,
    verbosity=2,
    scoring="neg_mean_squared_error",
    config_dict=tpot.config.regressor_config_dict,
    n_jobs=4,
    random_state=RANDOM_SEED
)
model.fit(features=train.drop(columns=regression_irrelevant + ['2dm_id (unitless)', 'formula']), target=train['bandgap (eV)'])


# In[]:


train


# In[ ]:




