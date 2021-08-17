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


# In[]:


RANDOM_SEED = 1234

pd.options.mode.chained_assignment = None
tqdm.tqdm.pandas()
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# In[]:


df = pd.read_pickle("../DigitalEcosystem/raw_data/2d_mat_dataset_raw.pkl")
df = df.drop(columns=['2dm_id (unitless)',
                      'formula',
                      'discovery_process (unitless)',
                      'potcars (unitless)',
                      'is_hubbard (unitless)',
                      'energy_per_atom (eV)',
                      'decomposition_energy (eV/atom)',
                      'is_bandgap_direct (unitless)',
                      'is_metal (unitless)',
                      'energy_vdw_per_atom (eV/atom)'])
df


# In[]:


structures = df['atoms_object (unitless)'].apply(pymatgen.io.ase.AseAtomsAdaptor.get_structure)
df['ox_struct'] = structures.apply(lambda i: i.copy())
df.ox_struct.progress_apply(lambda struct: struct.add_oxidation_state_by_guess())
df


# In[]:


df = df.drop(columns=['atoms_object (unitless)'])


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
    df['global_instability']=df.ox_struct.progress_apply(maybe_global_instability)
df


# In[]:


from pymatgen.analysis.local_env import JmolNN
neighbor_finder = JmolNN()


# In[]:


from matminer.featurizers.site.bonding import AverageBondLength

def average_bond_length(structure, featurizer = AverageBondLength(neighbor_finder)):
    n_atoms = len(structure)
    try:
        lengths = map(lambda i: featurizer.featurize(structure, i)[0], range(n_atoms))
        return sum(lengths) / n_atoms
    except IndexError:
        return None

df['bond_length_average'] = df.ox_struct.progress_apply(average_bond_length).copy()
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
df['bond_angle_average'] = df.ox_struct.progress_apply(average_bond_angle).copy()
df


# In[]:


def average_cn(structure, neighbor_finder = neighbor_finder):
    n_atoms = len(structure)
    cns = map(lambda i: neighbor_finder.get_cn(structure, i), range(n_atoms))
    return sum(cns) / n_atoms
df['average_cn'] = df.ox_struct.progress_apply(average_cn).copy()
df


# In[]:


def ab_perimeter_area_ratio(structure):
    a, b, c = structure.lattice.matrix
    perimeter = 2*np.linalg.norm(a) + 2*np.linalg.norm(b)
    area = np.linalg.norm(np.cross(a,b))
    return perimeter / area
df['perimeter_area_ratio'] = df.ox_struct.progress_apply(ab_perimeter_area_ratio).copy()
df


# In[]:


def slab_thickness(structure):
    c_coords = structure.cart_coords[:,2]
    thickness = max(c_coords) - min(c_coords)
    return thickness
df['slab_thickness'] = df.ox_struct.progress_apply(slab_thickness).copy()
df


# In[]:


data = df.drop(columns=['ox_struct', 'bandgap (eV)', 'total_magnetization (Bohr Magneton)'])
data


# In[]:


import tpot
import sklearn.model_selection
import sklearn.impute
imputer = sklearn.impute.KNNImputer(weights='distance')

data_train, data_test = sklearn.model_selection.train_test_split(data, test_size=0.25, random_state=RANDOM_SEED)
data_train = pd.DataFrame(imputer.fit_transform(data_train), columns=data_train.columns)
data_test = pd.DataFrame(imputer.transform(data_test), columns=data_test.columns)


# In[]:


data_train.to_csv("exfol_extraFeatures_kNNImputed_train.csv")
data_test.to_csv("exfol_extraFeatures_kNNImputed_test.csv")

target = 'exfoliation_energy_per_atom (eV/atom)'
x = data_train.drop(columns=[target]).to_numpy()
y = data_train[target].to_numpy()


# In[]:


model = tpot.TPOTRegressor(
    generations=None,
    population_size=100,
    max_eval_time_mins=10/60,
    max_time_mins=25,
    cv=5,
    verbosity=2,
    scoring="r2",
    config_dict=tpot.config.regressor_config_dict,
    n_jobs=8,
    random_state=RANDOM_SEED
)
model.fit(features=x, target=y)


# In[]:


train_y_true = y
train_y_pred = model.predict(x)

test_x = data_test.drop(columns=[target]).to_numpy()
test_y = data_test[target].to_numpy()

test_y_true = test_y
test_y_pred = model.predict(test_x)


# In[]:


plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
plt.rcParams["figure.figsize"] = (10,10)

plt.scatter(x=train_y_pred, y=train_y_true, label="Train Set")
plt.scatter(x=test_y_pred, y=test_y_true, label="Test Set")
plt.plot([0,2], [0,2], label="Parity")
plt.ylabel("Exfol (Actual)")
plt.xlabel("Exfol (Predicted)")
plt.legend()


# In[]:


tree=model.fitted_pipeline_[-1]


# In[]:


importances = zip(tree.feature_importances_, data_train.drop(columns=[target]).columns)


# In[]:


for i,j in sorted(importances, key=lambda i: -i[0]):
    print(i,j)


# In[]:


sisso_fun = (
    ("Rung1_1Term", lambda df: 9.416504816846741e-02 + 1.610701833162262e-03 * (df["min:evaporation_heat"] / df["min:electron_negativity"])),
    ("Rung1_2Term", lambda df: 7.889709146754061e-02 + 1.113430001456383e+00 * (df["var:num_p_unfilled"] / df["sum:num_valance"]) + 1.287409038211310e-03 * (df["min:evaporation_heat"] / df["min:electron_negativity"]))
)

for dataframe in (data_train, data_test):
    for name, fun in sisso_fun:
        dataframe[name] = fun(dataframe)


# In[]:


plt.rcParams["font.size"] = 16

tpot_mae =  np.round(sklearn.metrics.mean_absolute_error(y_true=test_y_true, y_pred=test_y_pred), 2)
r1_1t_mae = np.round(sklearn.metrics.mean_absolute_error(y_true=test_y_true, y_pred=data_test[sisso_fun[0][0]]),2)
r1_2t_mae = np.round(sklearn.metrics.mean_absolute_error(y_true=test_y_true, y_pred=data_test[sisso_fun[1][0]]),2)


fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True)
ax1.scatter(y=data_test["exfoliation_energy_per_atom (eV/atom)"],  x=data_test[sisso_fun[0][0]], marker="o",   color="tab:red", label=f"SISSO_R1_1T_exfol_Test, MAE={r1_1t_mae}")
ax2.scatter(y=data_test["exfoliation_energy_per_atom (eV/atom)"],  x=data_test[sisso_fun[1][0]], marker="o",   color="tab:olive",label=f"SISSO_R1_2T_exfol_Test, MAE={r1_2t_mae}")

for ax in (ax1, ax2):
    ax.set_box_aspect(1)
    ax.plot([-1,2], [-1,2], c="midnightblue", label="Parity")
    ax.scatter(x=test_y_pred, y=test_y_true, color="#11205B", label=f"TPOT, MAE={tpot_mae}")
    ax.legend()
    ax.set_xlabel("Predicted Exfoliation Energy (eV/atom)")
    ax.set_ylabel("Actual Exfolation Energy (eV/atom)")
    ax.set_title("Testing Set (25% holdout)")
fig.set_size_inches(20,20)


# In[]:


[i for i in df.columns]


# In[ ]:




