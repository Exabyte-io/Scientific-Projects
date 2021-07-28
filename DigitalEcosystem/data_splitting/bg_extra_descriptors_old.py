#!/usr/bin/env python
# coding: utf-8

# In[]:


import collections
import random

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import matminer
import pymatgen.core, pymatgen.core.composition
import pymatgen.io.ase

import pandas as pd
import swifter

RANDOM_SEED = 12345

pd.options.mode.chained_assignment = None
tqdm.tqdm.pandas()
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# In[]:


df = pd.read_pickle("../raw_data/2d_mat_dataset_raw.pkl")
df


# In[]:


structures = df['atoms_object (unitless)'].apply(pymatgen.io.ase.AseAtomsAdaptor.get_structure)
df['ox_struct'] = structures.apply(lambda i: i.copy())
#df.ox_struct.swifter.apply(lambda struct: struct.add_oxidation_state_by_guess())
df.ox_struct.swifter.apply(lambda struct: struct.add_oxidation_state_by_guess())
df


# In[]:


from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.structure.misc import EwaldEnergy
from matminer.featurizers.structure.order import StructuralComplexity
from matminer.featurizers.structure.symmetry import GlobalSymmetryFeatures

struct_features = MultipleFeaturizer([
    EwaldEnergy(),
    StructuralComplexity(),
    GlobalSymmetryFeatures('n_symmetry_ops'),
])
df[struct_features.feature_labels()] = struct_features.featurize_many(df.ox_struct)
df


# In[]:


import warnings
from matminer.featurizers.structure.bonding import GlobalInstabilityIndex
desc = GlobalInstabilityIndex()

def maybe_global_instability(struct):
    try:
        return desc.featurize(struct)[0]
    except:
        return None

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    df['global_instability']=df.ox_struct.swifter.apply(maybe_global_instability)
df


# In[]:


from pymatgen.analysis.local_env import JmolNN
from matminer.featurizers.site.bonding import AverageBondLength

neighbor_finder = JmolNN()

def average_bond_length(structure, featurizer = AverageBondLength(neighbor_finder)):
    n_atoms = len(structure)
    try:
        lengths = map(lambda i: featurizer.featurize(structure, i)[0], range(n_atoms))
        return sum(lengths) / n_atoms
    except IndexError:
        return None

df['bond_length_average'] = df.ox_struct.swifter.apply(average_bond_length).copy()
df


# In[]:


from matminer.featurizers.site.bonding import AverageBondAngle

def average_bond_angle(structure, featurizer = AverageBondAngle(neighbor_finder)):
    n_atoms = len(structure)
    try:
        angles = map(lambda i: featurizer.featurize(structure, i)[0], range(n_atoms))
        return sum(angles) / n_atoms
    except IndexError:
        return None
df['bond_angle_average'] = df.ox_struct.swifter.apply(average_bond_angle).copy()
df


# In[]:


def average_cn(structure, neighbor_finder = neighbor_finder):
    n_atoms = len(structure)
    cns = map(lambda i: neighbor_finder.get_cn(structure, i), range(n_atoms))
    return sum(cns) / n_atoms
df['average_cn'] = df.ox_struct.swifter.apply(average_cn).copy()
df


# In[]:


def ab_perimeter_area_ratio(structure):
    a, b, c = structure.lattice.matrix
    perimeter = 2*np.linalg.norm(a) + 2*np.linalg.norm(b)
    area = np.linalg.norm(np.cross(a,b))
    return perimeter / area
df['perimeter_area_ratio'] = df.ox_struct.swifter.apply(ab_perimeter_area_ratio).copy()
df


# In[]:


df["formula"] = df["atoms_object (unitless)"].swifter.apply(lambda atoms: atoms.get_chemical_formula(empirical=True))
df


# In[]:


# df = df_bak.copy()
# df_bak.to_pickle("all_descriptors.pkl")


# In[]:


#df = df[df["discovery_process (unitless)"] == "top-down"]
#df.to_pickle("all_descriptors_topdown.pkl")
to_drop = [ "discovery_process (unitless)",
            "atoms_object (unitless)",
            'potcars (unitless)',
            'is_hubbard (unitless)',
            'energy_per_atom (eV)',
            'decomposition_energy (eV/atom)',
            'is_bandgap_direct (unitless)',
            'is_metal (unitless)',
            'energy_vdw_per_atom (eV/atom)',
            'total_magnetization (Bohr Magneton)',
            'ox_struct']
df.drop(columns=to_drop).to_csv("bg_extra_descriptors_topdown.csv")
df


# # Chemical Info

# In[]:


#df_physical_aphysical = df.copy()
df = df[df["exfoliation_energy_per_atom (eV/atom)"].apply(lambda exfol: 0 <= exfol <= 0.1) & df["decomposition_energy (eV/atom)"] == 0]


# In[]:


df[df["decomposition_energy (eV/atom)"] >= 0].drop(columns=to_drop).to_csv("both_topdown_decompGEQZero.csv")


# In[]:


df[df["exfoliation_energy_per_atom (eV/atom)"] <= 0].drop(columns=to_drop).to_csv("both_topdown_exfolLEQZero.csv")


# In[]:


df[df["is_hubbard (unitless)"] == True].drop(columns=to_drop).to_csv("both_topdown_transitionMetalOxidesFluorides.csv")


# In[]:


df[df["atoms_object (unitless)"].swifter.apply(lambda atoms: "N" in atoms.get_chemical_symbols())].drop(columns=to_drop).to_csv("both_topdown_nitridesOnly.csv")
df[df["atoms_object (unitless)"].swifter.apply(lambda atoms: "O" in atoms.get_chemical_symbols())].drop(columns=to_drop).to_csv("both_topdown_oxidesOnly.csv")
df[df["atoms_object (unitless)"].swifter.apply(lambda atoms: "S" in atoms.get_chemical_symbols())].drop(columns=to_drop).to_csv("both_topdown_sulfidesOnly.csv")
df[df["atoms_object (unitless)"].swifter.apply(lambda atoms: "C" in atoms.get_chemical_symbols())].drop(columns=to_drop).to_csv("both_topdown_carbidesOnly.csv")
df[df["atoms_object (unitless)"].swifter.apply(lambda atoms: any(symbol in ["F", "Cl", "Br", "Cl"] for symbol in atoms.get_chemical_symbols()))].drop(columns=to_drop).to_csv("both_topdown_halidesOnly.csv")
df[df["atoms_object (unitless)"].swifter.apply(lambda atoms: any(symbol in ["O", "S", "Se", "Te"] for symbol in atoms.get_chemical_symbols()))].drop(columns=to_drop).to_csv("both_topdown_chaclogensOnly.csv")
df[df["atoms_object (unitless)"].swifter.apply(lambda atoms: any(symbol in ["N", "P", "As", "Sb"] for symbol in atoms.get_chemical_symbols()))].drop(columns=to_drop).to_csv("both_topdown_pnictogensOnly.csv")


# In[]:


pblock_metals = ["Al", "Ga", "Ge", "In", "Sn", "Sb", "Tl", "Pb", "Bi", "Po"]
df[df["atoms_object (unitless)"].swifter.apply(lambda atoms: any(symbol in pblock_metals for symbol in atoms.get_chemical_symbols()))].drop(columns=to_drop).to_csv("both_topdown_pBlockMetalsOnly.csv")


# In[]:


d9_coinage_metals = ["Cu", "Ag", "Au"]
df[df["atoms_object (unitless)"].swifter.apply(lambda atoms: any(symbol in d9_coinage_metals for symbol in atoms.get_chemical_symbols()))].drop(columns=to_drop).to_csv("both_topdown_d9CoinageMetalsOnly.csv")


# In[]:


# Based on https://en.wikipedia.org/wiki/Wide-bandgap_semiconductor and https://www.kymatech.com/about/faqs/429-what-is-an-ultra-wide-bandgap-semiconductor-material
df[df["bandgap (eV)"] == 0].drop(columns=to_drop).to_csv("both_topdown_metallicOnly.csv")
df[df["bandgap (eV)"].swifter.apply(lambda bg: 0 < bg < 2)].drop(columns=to_drop).to_csv("both_topdown_conventionalSemiconductorsOnly.csv")
df[df["bandgap (eV)"].swifter.apply(lambda bg: 2 <= bg <= 3.4)].drop(columns=to_drop).to_csv("both_topdown_WBGOnly.csv")
df[df["bandgap (eV)"].swifter.apply(lambda bg: 3.4 < bg < 10)].drop(columns=to_drop).to_csv("both_topdown_UWBGOnly.csv")


# # KNN

# In[]:


import sklearn.cluster


# In[]:


n_clusters = 8
kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters,
                                max_iter=3000,
                                random_state=RANDOM_SEED)
data_kmeans = df.drop(columns=to_drop + ["formula", "2dm_id (unitless)"]).fillna(df.drop(columns=to_drop + ["formula", "2dm_id (unitless)"]).mean())
classes = kmeans.fit_predict(data_kmeans.to_numpy())
df["kmeans_cluster"] = classes

for i in range(n_clusters):
    df[df["kmeans_cluster"]==i].drop(columns=to_drop + ["kmeans_cluster"]).to_csv(f"both_topdown_cluster{i}.csv")


# In[]:


df


# In[]:


df[df["atoms_object (unitless)"].swifter.apply(lambda atoms: "C" in atoms.get_chemical_symbols())]


# In[]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[]:





# In[]:


plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
plt.rcParams["figure.figsize"] = (10,10)
decomps = df["exfoliation_energy_per_atom (eV/atom)"][df["exfoliation_energy_per_atom (eV/atom)"].apply(lambda exfol: 0 < exfol < 1)].to_numpy()
sns.histplot(decomps, bins=200)


# In[]:


for i in [0.25, 0.5, 0.75]:
    print(i, df["exfoliation_energy_per_atom (eV/atom)"].quantile(i))
# df["exfoliation_energy_per_atom (eV/atom)"].describeibe()


# In[]:


df


# In[]:


nonmetal = df[df["bandgap (eV)"]!=0].drop(columns=to_drop + ["2dm_id (unitless)", "formula", "exfoliation_energy_per_atom (eV/atom)", "kmeans_cluster"])


# In[]:


colmask = [":" not in i for i in nonmetal.columns]
masked_df = nonmetal[nonmetal.columns[colmask]]


# In[]:


import sklearn.model_selection
import tpot

data_train, data_test = sklearn.model_selection.train_test_split(nonmetal, test_size=0.25, random_state=RANDOM_SEED)
train_x = data_train.drop(columns="bandgap (eV)").to_numpy()
train_y = data_train["bandgap (eV)"].to_numpy()

test_x = data_test.drop(columns="bandgap (eV)").to_numpy()
test_y = data_test["bandgap (eV)"].to_numpy()


# In[]:


model = tpot.TPOTRegressor(
    generations=None,
    population_size=100,
    max_eval_time_mins=5/60,
    max_time_mins=1,
    cv=5,
    verbosity=2,
    scoring="neg_mean_squared_error",
    config_dict=tpot.config.regressor_config_dict,
    n_jobs=4,
    random_state=RANDOM_SEED
)
model.fit(features=train_x, target=train_y)


# In[]:


train_y_pred = model.predict(train_x)
test_y_pred = model.predict(test_x)

plt.scatter(x=train_y_pred, y=train_y, label="Train Set")
plt.scatter(x=test_y_pred, y=test_y, label="Test Set")
plt.plot([0,8], [0,8], label="Parity")
plt.ylabel("Bandgap (Actual)")
plt.xlabel("Bandgap (Predicted)")
plt.legend()


# In[]:


g = sns.pairplot(masked_df)
g.map_lower(sns.kdeplot, levels=4, color=".2")


# In[]:





# In[]:





# In[]:


temp = df[df["bandgap (eV)"]>0.1].copy()

x = temp.drop(columns=to_drop + [ "exfoliation_energy_per_atom (eV/atom)", "kmeans_cluster", "bandgap (eV)", "formula", "2dm_id (unitless)"])
y = temp["bandgap (eV)"]

y_pred = model.predict(x)

temp["pred"] = y_pred
temp["abs_err"] = abs(temp["pred"] - temp["bandgap (eV)"]) 
temp["tan_err"] = temp["pred"] / temp["bandgap (eV)"]
temp = temp[temp["decomposition_energy (eV/atom)"] == 0]

fblock_and_synth = ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
                    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
                    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Tg", "Cn", "Nh", "Fl", "Mc", "Mc", "Lv", "Ts", "Og"]
temp = temp[temp["atoms_object (unitless)"].apply(lambda atoms: all(symbol not in fblock_and_synth for symbol in atoms.get_chemical_symbols()))].sort_values("tan_err")
temp[temp["is_hubbard (unitless)"]==True]


# In[]:


df["decomposition_energy (eV/atom)"].describe()


# In[]:


import sklearn.model_selection
import tpot

temp = temp.drop(columns=["pred", "abs_err", "tan_err"])


# In[]:


data_train, data_test = sklearn.model_selection.train_test_split(temp, test_size=0.1, random_state=RANDOM_SEED)
train_x = data_train.drop(columns="bandgap (eV)").to_numpy()
train_y = data_train["bandgap (eV)"].to_numpy()

test_x = data_test.drop(columns="bandgap (eV)").to_numpy()
test_y = data_test["bandgap (eV)"].to_numpy()

model = tpot.TPOTRegressor(
    generations=None,
    population_size=120,
    max_eval_time_mins=5/60,
    max_time_mins=1,
    cv=4,
    verbosity=2,
    scoring="neg_root_mean_squared_error",
    config_dict=tpot.config.regressor_config_dict,
    n_jobs=6,
    random_state=RANDOM_SEED
)
model.fit(features=train_x, target=train_y)


# In[]:


train_y_pred = model.predict(train_x)
test_y_pred = model.predict(test_x)

plt.scatter(x=train_y_pred, y=train_y, label="Train Set")
plt.scatter(x=test_y_pred, y=test_y, label="Test Set")
plt.plot([0,8], [0,8], label="Parity")
plt.ylabel("Bandgap (Actual)")
plt.xlabel("Bandgap (Predicted)")
plt.legend()


# In[]:


sklearn.metrics.mean_absolute_error(test_y, test_y_pred)


# In[]:


foo = sklearn.model_selection.cross_validate(model.fitted_pipeline_, temp.drop(columns="bandgap (eV)").fillna(temp.drop(columns="bandgap (eV)").mean()).to_numpy(),
                                             temp["bandgap (eV)"],
                                             scoring="neg_mean_absolute_error")


# In[]:


abs(sum(foo["test_score"]) / len(foo["test_score"]))


# In[ ]:





# In[ ]:




