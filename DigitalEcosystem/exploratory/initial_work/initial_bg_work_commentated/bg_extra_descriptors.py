#!/usr/bin/env python
# coding: utf-8

# # Initial Bandgap Screening
# 
# In this notebook, we conduct some initial, primarily exploratory work to predict the bandgap of 2D materials using both compositional and structural features.

# In[]:


import collections
import warnings
import random

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import tpot
import sklearn.model_selection
import sklearn.impute
import pymatgen.core, pymatgen.core.composition
import pymatgen.io.ase
from pymatgen.analysis.local_env import JmolNN
import matminer
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.structure.misc import EwaldEnergy
from matminer.featurizers.structure.order import StructuralComplexity
from matminer.featurizers.structure.symmetry import GlobalSymmetryFeatures
from matminer.featurizers.structure.bonding import GlobalInstabilityIndex
from matminer.featurizers.site.bonding import AverageBondLength
from matminer.featurizers.site.bonding import AverageBondAngle


import pandas as pd


# In[]:


RANDOM_SEED = 1234

pd.options.mode.chained_assignment = None
tqdm.tqdm.pandas()
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# In[ ]:


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


# # Featurize the Dataset
# 
# Now, we're going to add features to our dataset. We already have several compositional features calculated using
# XenonPy, and we may get some better performance out of our models if we also incorporate structural descriptors.
# 
# We'll start by leveraging PyMatGen to estimate the oxidation state of our 2D materials. This will be useful for the
# calculation of some of our features down the road, which require information about the oxidation state.

# In[ ]:


structures = df['atoms_object (unitless)'].apply(pymatgen.io.ase.AseAtomsAdaptor.get_structure)
df['ox_struct'] = structures.apply(lambda i: i.copy())
df.ox_struct.progress_apply(lambda struct: struct.add_oxidation_state_by_guess())


# Now that we don't need the ASE atoms object anymore, we can drop it from our dataset.

# In[]:


df = df.drop(columns=['atoms_object (unitless)'])


# ## Ewald Energy
# [Documentation Link](https://hackingmaterials.lbl.gov/matminer/matminer.featurizers.structure.html#matminer.featurizers.structure.misc.EwaldEnergy)
# 
# Using the partial charges we estimated earlier, this will approximate the long-range interactions between ions in our
# 2D systems.
# 
# ## Structural Complexity
# [Documentation Link](https://hackingmaterials.lbl.gov/matminer/matminer.featurizers.structure.html#matminer.featurizers.structure.order.StructuralComplexity)
# 
# This is a variation on the Shannon entropy that accounts for the symmetry of the unit cell. This helps assess how
# organized / disorganized our system is, providing some estimation of its entropy.
# 
# ## N Symmetry Ops
# [Documentation Link](https://hackingmaterials.lbl.gov/matminer/matminer.featurizers.structure.html#matminer.featurizers.structure.order.StructuralComplexity)
# 
# Number of symmetry operations allowed by the point group in our unit cell.

# In[ ]:


struct_features = MultipleFeaturizer([
    EwaldEnergy(),
    StructuralComplexity(),
    GlobalSymmetryFeatures('n_symmetry_ops'),
])
df[struct_features.feature_labels()] = struct_features.featurize_many(df.ox_struct)


# ## Global Instability Index
# [Documentation Link](https://hackingmaterials.lbl.gov/matminer/matminer.featurizers.structure.html#matminer.featurizers.structure.order.StructuralComplexity)
# 
# The global instability index is based on the work of Salinas-Sanches, A. in the following publication: [Link](https://www.sciencedirect.com/science/article/abs/pii/002245969290094C?via%3Dihub).
# 
# Essentially, it estimates, on average, how strained the bonds are in the system (e.g. are the over-or under-bonding).
# 

# In[ ]:


desc = GlobalInstabilityIndex()

def maybe_global_instability(struct):
    try:
        return desc.featurize(struct)[0]
    except:
        return None

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    df['global_instability']=df.ox_struct.progress_apply(maybe_global_instability)


# ## Choosing a Neighbor Finder
# 
# The next several descriptors will require us to define what exactly it means for two atoms to be "neighbors." Many of
# the neighbor-finders in PyMatGen use a voronoi-based algorithm that has trouble with some slabs. As of August 2021,
# the issue is still open, as shown in this link: [Link](https://github.com/materialsproject/pymatgen/issues/801)
# 
# Here, we'll take the neighbor-finding algorithm of JMol, as implemented in PyMatGen.

# In[]:


neighbor_finder = JmolNN()


# ## Average Bond Length
# [Documentation Link](https://hackingmaterials.lbl.gov/matminer/matminer.featurizers.site.html#matminer.featurizers.site.bonding.AverageBondLength)
# 
# We will also calculate the average bond length in our system.

# In[ ]:


def average_bond_length(structure, featurizer = AverageBondLength(neighbor_finder)):
    n_atoms = len(structure)
    try:
        lengths = map(lambda i: featurizer.featurize(structure, i)[0], range(n_atoms))
        return sum(lengths) / n_atoms
    except IndexError:
        return None

df['bond_length_average'] = df.ox_struct.progress_apply(average_bond_length).copy()


# ## Average Bond Angle
# [Documentation Link](https://hackingmaterials.lbl.gov/matminer/matminer.featurizers.site.html#matminer.featurizers.site.bonding.AverageBondAngle)
# 
# We'll additionally determine the average bond angle between atoms in our system.

# In[ ]:


def average_bond_angle(structure, featurizer = AverageBondAngle(neighbor_finder)):
    n_atoms = len(structure)
    try:
        angles = map(lambda i: featurizer.featurize(structure, i)[0], range(n_atoms))
        return sum(angles) / n_atoms
    except IndexError:
        return None
df['bond_angle_average'] = df.ox_struct.progress_apply(average_bond_angle).copy()


# ## Average Coordination Number
# 
# We'll determine how coordinated, on average, the different atoms in our system are. This is accomplished by
# determining the number of nearest neighbors with the neighbor finder we defined earlier, and averaging over
# the atoms in the crystal cell.

# In[ ]:


def average_cn(structure, neighbor_finder = neighbor_finder):
    n_atoms = len(structure)
    cns = map(lambda i: neighbor_finder.get_cn(structure, i), range(n_atoms))
    return sum(cns) / n_atoms
df['average_cn'] = df.ox_struct.progress_apply(average_cn).copy()


# ## AB Perimeter/Area Ratio
# 
# Next, we'll try to assess how square our unit cell is in the two directions parallel to the surface. To do this, we'll
# take the ratio of the perimeter to the area of the cell.

# In[ ]:


def ab_perimeter_area_ratio(structure):
    a, b, c = structure.lattice.matrix
    perimeter = 2*np.linalg.norm(a) + 2*np.linalg.norm(b)
    area = np.linalg.norm(np.cross(a,b))
    return perimeter / area
df['perimeter_area_ratio'] = df.ox_struct.progress_apply(ab_perimeter_area_ratio).copy()


# ## Slab Thickness
# 
# Finally, we'll assess how thick the slab we're looking at is, since some of the slabs are rather thick.

# In[ ]:


def slab_thickness(structure):
    c_coords = structure.cart_coords[:,2]
    thickness = max(c_coords) - min(c_coords)
    return thickness
df['slab_thickness'] = df.ox_struct.progress_apply(slab_thickness).copy()


# # Final Data prep
# 
# Next, we'll drop the column we created to store Pymatgen's estimated oxidation states (`ox_struct`).
# 
# Because we are focusing on the bandgap without requiring the use of DFT, we'll also drop the `exfoliation_energy_per_atom`
# and `total_magnetization` columns, which were both DFT-calculated properties.

# In[]:


data = df.drop(columns=['ox_struct', 'exfoliation_energy_per_atom (eV/atom)', 'total_magnetization (Bohr Magneton)'])
data


# There may be some missing values for some of the above features, so we'll also do some imputation via KNN for these values.
# 
# We'll then take a train/test split, holding out 25% of the data as a testing set.
# 
# Note that we need to be careful to avoid test-set leakage here. Because the KNN imputer fits a KNN model, we make sure
# that it is fit using only the training set. The testing set then uses the K neighbors identified on the training set.

# In[]:


imputer = sklearn.impute.KNNImputer(weights='distance')

data_train, data_test = sklearn.model_selection.train_test_split(data, test_size=0.25, random_state=RANDOM_SEED)
data_train = pd.DataFrame(imputer.fit_transform(data_train), columns=data_train.columns)
data_test = pd.DataFrame(imputer.transform(data_test), columns=data_test.columns)


# Finally, we'll write the training and testing set to a CSV.  At this point, we took the CSV files and ran SISSO on them,
# on the cluster. To have something to compare to (without worrying too much about tuning a pipeline on our own), we
# just use TPOT here.

# In[]:


data_train.to_csv("bg_extraFeatures_kNNImputed_train.csv")
data_test.to_csv("bg_extraFeatures_kNNImputed_test.csv")

target = 'bandgap (eV)'
x = data_train.drop(columns=[target]).to_numpy()
y = data_train[target].to_numpy()


# # TPOT
# 
# TPOT is a genetic algorithm designed to find pipelines that fit a data. To avoid overfitting, it internally uses a
# cross-validation strategy. At this time, because we were looking for *any* correlation with the dataset (because we
# had not yet had good results), we set TPOT to optimize for the R2.

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
    n_jobs=4,
    random_state=RANDOM_SEED
)
model.fit(features=x, target=y)


# ## TPOT Plot
# Next, we will plot the results of our TPOT run. Our selected model was an Extra Trees regressor, from Scikit-Learn.
# 

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
plt.plot([0,8], [0,8], label="Parity")
plt.ylabel("Bandgap (Actual)")
plt.xlabel("Bandgap (Predicted)")
plt.legend()


# Next, we'll get the importances of that were found by the tree in the resultant TPOT model.

# In[]:


tree=model.fitted_pipeline_[0]
importances = list(zip(tree.feature_importances_, data_train.drop(columns=[target]).columns))


# In[]:


for i,j in sorted(importances, key=lambda i: -i[0]):
    print(i,j)


# # Comparison to SISSO
# 
# SISSO results in a few different models, at Rung 1 and Rung 2. 
# 
# ## Rung 1 (Full Feature Set)
# Because of the current scaling, we're restricted to rung 1 in the case of our full feature set.

# In[]:


sisso_fun = (
    ("Rung1_1Term", lambda df: -5.446832075317974e-01 + 1.523419105194092e-01 * (df["min:gs_energy"] + df["max:en_allen"])),
    ("Rung1_2Term", lambda df: 4.094491794919684e+00 + -1.023239993380321e-01 * (df["ave:covalent_radius_cordero"] / df["ave:gs_est_bcc_latcnt"]) + 8.702895980630683e+02 * (df["var:en_pauling"] / df["ave:boiling_point"]))
)

for dataframe in (data_train, data_test):
    for name, fun in sisso_fun:
        dataframe[name] = fun(dataframe)


# Plot the results of the SISSO runs.

# In[]:


plt.rcParams["font.size"] = 16

tpot_mae =  np.round(sklearn.metrics.mean_absolute_error(y_true=test_y_true, y_pred=test_y_pred), 2)
r1_1t_mae = np.round(sklearn.metrics.mean_absolute_error(y_true=test_y_true, y_pred=data_test[sisso_fun[0][0]]),2)
r1_2t_mae = np.round(sklearn.metrics.mean_absolute_error(y_true=test_y_true, y_pred=data_test[sisso_fun[1][0]]),2)


fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True)
ax1.scatter(y=data_test["bandgap (eV)"],  x=data_test[sisso_fun[0][0]], marker="o",   color="tab:red", label=f"SISSO_R1_1T_BG_Test, MAE={r1_1t_mae}")
ax2.scatter(y=data_test["bandgap (eV)"],  x=data_test[sisso_fun[1][0]], marker="o",   color="tab:olive",label=f"SISSO_R1_2T_Test, MAE={r1_2t_mae}")

for ax in (ax1, ax2):
    ax.set_box_aspect(1)
    ax.plot([0,8], [0,8], c="black", label="Parity")
    ax.scatter(x=test_y_pred, y=test_y_true, color="midnightblue", label=f"TPOT, MAE={tpot_mae}")
    ax.legend()
    ax.set_xlabel("Predicted Bandgap (eV)")
    ax.set_ylabel("Actual Bandgap (eV)")
    ax.set_title("Testing Set (25% holdout)")
fig.set_size_inches(20,20)


# ## Rung 2 (Subset of Features)
# 
# Next, to get to rung 2, we'll take a subset of the chosen features. Specifically, we'll do choose the structural features we've generated, along with the 10 most-important features identified in TPOT's extra-trees regressor.

# In[]:


subset_features = ["structural complexity per atom",
                   "structural complexity per cell",
                   "n_symmetry_ops",
                   "global_instability",
                   "bond_length_average",
                   "bond_angle_average",
                   "average_cn",
                   "perimeter_area_ratio",
                   "slab_thickness",
                   "bandgap (eV)"]
for count, (i,j) in enumerate(sorted(importances, key=lambda i: -i[0])):
    if count == 10:
        break
    subset_features.append(j)
higher_train = data_train[subset_features]
higher_test = data_test[subset_features]
higher_train.to_csv("higher_bg_train.csv")
higher_test.to_csv("higher_bg_test.csv")


# Next, define the models that had been generated via SISSO, and add them to our dataset.

# In[]:


higher_sisso_fun = (
    ("Rung1_1Term", lambda df: -5.694579741645713e-01 + 1.558078357170703e-01 * (df["max:en_allen"] + df["min:gs_energy"])),
    ("Rung1_2Term", lambda df: -1.033777408625130e+00 + 2.081460140480539e-01 * (df["max:en_pauling"] / df["perimeter_area_ratio"]) + 1.263751433179756e-01 * (df["max:en_allen"] + df["min:gs_energy"])),
    ("Rung2_1Term", lambda df: -1.256606521939596e+00 + -9.068477655636401e-01 * ((df["max:en_pauling"] / df["min:gs_energy"]) - (abs(df["max:en_pauling"] - df["perimeter_area_ratio"])))),
    ("Rung2_2Term", lambda df: -5.538612447174948e-01 + -3.982366078899126e+00 * ((df["ewald_energy_per_atom"] / df["min:boiling_point"]) / (df["perimeter_area_ratio"] + df["average_cn"])) + -5.802291194183285e+00 * ((abs(df["max:en_pauling"] - df["perimeter_area_ratio"])) / (df["min:gs_energy"] - df["bond_length_average"])))
)

for dataframe in (higher_train, higher_test):
    for name, fun in higher_sisso_fun:
        print(name)
        dataframe[name] = fun(dataframe)


# Finally, plot the Rung 2 results from SISSO.

# In[]:


plt.rcParams["font.size"] = 16

tpot_mae =  np.round(sklearn.metrics.mean_absolute_error(y_true=test_y_true, y_pred=test_y_pred), 2)
r1_1t_mae = np.round(sklearn.metrics.mean_absolute_error(y_true=test_y_true, y_pred=higher_test[higher_sisso_fun[0][0]]),2)
r1_2t_mae = np.round(sklearn.metrics.mean_absolute_error(y_true=test_y_true, y_pred=higher_test[higher_sisso_fun[1][0]]),2)
r2_1t_mae = np.round(sklearn.metrics.mean_absolute_error(y_true=test_y_true, y_pred=higher_test[higher_sisso_fun[2][0]]),2)
r2_2t_mae = np.round(sklearn.metrics.mean_absolute_error(y_true=test_y_true, y_pred=higher_test[higher_sisso_fun[3][0]]),2)



fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, sharex=True, sharey=True)
ax1.scatter(y=data_test["bandgap (eV)"],  x=higher_test[higher_sisso_fun[0][0]], marker="o",   color="tab:red",  label=f"SISSO_R1_1T_BG_Test, MAE={r1_1t_mae}")
ax2.scatter(y=data_test["bandgap (eV)"],  x=higher_test[higher_sisso_fun[1][0]], marker="o",   color="tab:olive",label=f"SISSO_R1_2T_Test, MAE={r1_2t_mae}")
ax3.scatter(y=data_test["bandgap (eV)"],  x=higher_test[higher_sisso_fun[2][0]], marker="o",   color="tab:green",label=f"SISSO_R2_1T_Test, MAE={r2_1t_mae}")
ax4.scatter(y=data_test["bandgap (eV)"],  x=higher_test[higher_sisso_fun[3][0]], marker="o",   color="tab:brown",label=f"SISSO_R2_2T_Test, MAE={r2_2t_mae}")


for ax in (ax1, ax2, ax3, ax4):
    ax.set_box_aspect(1)
    ax.plot([0,8], [0,8], c="black", label="Parity")
    ax.scatter(x=test_y_pred, y=test_y_true, color="midnightblue", label=f"TPOT, MAE={tpot_mae}")
    ax.legend()
    ax.set_xlabel("Predicted Bandgap (eV)")
    ax.set_ylabel("Actual Bandgap (eV)")
    ax.set_title("Testing Set (25% holdout)")
fig.set_size_inches(40,20)


# In[ ]:




