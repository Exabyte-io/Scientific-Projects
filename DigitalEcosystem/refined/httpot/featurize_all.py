#!/usr/bin/env python
# coding: utf-8

# # Featurization
#
# In this notebook, we apply some structural descriptors to our dataset, such as the number of `halide`-`transitoin metal`-`halide` bonds in the system. These are a bit expensive to calculate, and end up taking an hour or two. For this reason, we pickle them here.

# In[]:


import itertools
import warnings
import collections

import pandas as pd
import random
import numpy as np

import tqdm

import pymatgen.io
from pymatgen.analysis.local_env import JmolNN

from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.structure.misc import EwaldEnergy
from matminer.featurizers.structure.order import StructuralComplexity
from matminer.featurizers.structure.symmetry import GlobalSymmetryFeatures

# In[]:
from DigitalEcosystem.utils.fingerprints import maybe_global_instability, average_bond_length, average_bond_angle, \
    average_cn, ab_perimeter_area_ratio, neighbor_finder

RANDOM_SEED = 1234
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
pd.options.mode.chained_assignment = None
tqdm.tqdm.pandas()


# # Read the Datset
#
# We'll start by loading up the entire dataset.

# In[]:


dataset_path = "../../raw_data/2d_mat_dataset_raw.pkl"
df = pd.read_pickle(dataset_path)


# # Featurize the Dataset
#
# Now, we're going to add features to our dataset. We already have several compositional features calculated using
# XenonPy, and we may get some better performance out of our models if we also incorporate structural descriptors.
#
# We'll start by leveraging PyMatGen to estimate the oxidation state of our 2D materials. This will be useful for the
# calculation of some of our features down the road, which require information about the oxidation state.

# In[]:


# Pymatgen oxidation structure guess (needed for later)

structures = df['atoms_object (unitless)'].apply(pymatgen.io.ase.AseAtomsAdaptor.get_structure)
df['ox_struct'] = structures.apply(lambda i: i.copy())

# struct.add_oxidation_state_by_guess() modifies the structure in-place
df.ox_struct.swifter.apply(lambda struct: struct.add_oxidation_state_by_guess())


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

# In[]:


# Ewald Energy, Structural Complexity, and Global Symmetry Indices

struct_features = MultipleFeaturizer([
    EwaldEnergy(),
    StructuralComplexity(),
    GlobalSymmetryFeatures('n_symmetry_ops'),
])
df[struct_features.feature_labels()] = struct_features.featurize_many(df.ox_struct).copy()
print(struct_features.feature_labels())


# ## Global Instability Index
# [Documentation Link](https://hackingmaterials.lbl.gov/matminer/matminer.featurizers.structure.html#matminer.featurizers.structure.order.StructuralComplexity)
#
# The global instability index is based on the work of Salinas-Sanches, A. in the following publication: [Link](https://www.sciencedirect.com/science/article/abs/pii/002245969290094C?via%3Dihub).
#
# Essentially, it estimates, on average, how strained the bonds are in the system (e.g. are the over-or under-bonding).
#

# In[]:


with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    df['global_instability'] = df.ox_struct.swifter.apply(maybe_global_instability).copy()


# ## Choosing a Neighbor Finder
#
# The next several descriptors will require us to define what exactly it means for two atoms to be "neighbors." Many of
# the neighbor-finders in PyMatGen use a voronoi-based algorithm that has trouble with some slabs. As of August 2021,
# the issue is still open, as shown in this link: [Link](https://github.com/materialsproject/pymatgen/issues/801)
#
# Here, we'll take the neighbor-finding algorithm of JMol, as implemented in PyMatGen.

# In[]:


# ## Average Bond Length
# [Documentation Link](https://hackingmaterials.lbl.gov/matminer/matminer.featurizers.site.html#matminer.featurizers.site.bonding.AverageBondLength)
#
# We will also calculate the average bond length in our system.

# In[]:


df['bond_length_average'] = df.ox_struct.swifter.apply(average_bond_length).copy()


# ## Average Bond Angle
# [Documentation Link](https://hackingmaterials.lbl.gov/matminer/matminer.featurizers.site.html#matminer.featurizers.site.bonding.AverageBondAngle)
#
# We'll additionally determine the average bond angle between atoms in our system.

# In[]:


df['bond_angle_average'] = df.ox_struct.swifter.apply(average_bond_angle).copy()


# ## Average Coordination Number
#
# We'll determine how coordinated, on average, the different atoms in our system are. This is accomplished by
# determining the number of nearest neighbors with the neighbor finder we defined earlier, and averaging over
# the atoms in the crystal cell.

# In[]:


df['average_cn'] = df.ox_struct.swifter.apply(average_cn).copy()


# ## AB Perimeter/Area Ratio
#
# Next, we'll try to assess how square our unit cell is in the two directions parallel to the surface. To do this, we'll
# take the ratio of the perimeter to the area of the cell.

# In[]:


df['perimeter_area_ratio'] = df.ox_struct.swifter.apply(ab_perimeter_area_ratio).copy()


# Next, we'll extract the formula from the system.

# In[]:


df["formula"] = df["atoms_object (unitless)"].swifter.apply(lambda atoms: atoms.get_chemical_formula(empirical=True))


# ## Structural Descriptors
#
# Next, we'll calculate some structural descriptors, similar to the ones that had been used by Bhowmik, R. et al in [DOI 10.1016/j.polymer.2021.123558](https://doi.org/10.1016/j.polymer.2021.123558).
#
# Because there are a wide variety of elements in our dataset, to try and keep this from generating too many columns, we'll bin each element into different sections of the periodic table:
# - Alkaline metals
# - Alkaline earth metals
# - Early transition metals (d5 metals and earlier)
# - Late transition metals (d6 metals and later)
# - Triels (group 13 / Boron group)
# - Tetrels (group 14 / Carbon group)
# - Pnictogens
# - Chalcogens
# - Halides
# - Noble Gases
# - F-block elements
# - Post-Uranium elements
#
# We'll be looking for the following counts:
# - Number of each group (e.g. number of alkaline, number of triel, etc)
# - Number of each group-group bond (e.g. number of alkaline-early_transition, number of chalcogen-chalcogen, etc)
# - Number of each grou-grou-group angle (e.g. number of halide-late_transition-halide, number of tetrel-tetrel-tetrel, etc)
#
# Adding dihedrals as well would have required a more complex graph traversal algorithm, so in the interest of time we forego this.

# In[]:


classes = {
    'alkaline' : ['H', 'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr'],
    'alkaine_earth' : ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra'],
    'early_transition' : ['Sc', 'Ti', 'V', 'Cr', 'Mn',
                          'Y',  'Zr', 'Nb', 'Mo', 'Tc',
                                'Hf', 'Ta', 'W', 'Re',],
    'late_transition' : ['Fe', 'Co', 'Ni', 'Cu', 'Zn',
                         'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                         'Os', 'Ir', 'Pt', 'Au', 'Hg'],
    'triel' : ['B', 'Al', 'Ga', 'In', 'Tl'],
    'tetrel' : ['C', 'Si', 'Ge', 'Sn', 'Pb'],
    'pnictogen' : ['N', 'P', 'As', 'Sb', 'Bi'],
    'chalcogen' : ['O', 'S', 'Se', 'Te', 'Po'],
    'halide' : ['F', 'Cl', 'Br', 'I', 'At'],
    'noble_gas' : ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn'],
    'f_block' : ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                 'Ac', 'Th', 'Pa', 'U'],
    'post_uranium' : ['Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
                      'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
}

groups = {}
for key, values in classes.items():
    for val in values:
        groups[val] = key


# We then take a second pass over our system - this is probably rather inefficient. But this was faster to code.

# In[]:


symbols_cols = collections.Counter()
bond_cols = collections.Counter()
angle_cols = collections.Counter()

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


# # Back These Up
#
# This takes long enough that, to save progress, we save it.
#
# We then store the above run as a pickle. In this notebook, we can see a typo was present in its first version (`df.to_picke` instead of `df.to_pickle` had been called). To fix this, we re-ran that line in the below cell. The above featurization takes a couple of hours to run, so we didn't re-run the entire cell just for that.

# In[]:


df.to_pickle('backup.pkl')


# # Featurize the Data
#
# Next, we'll take the second pass at calculating these features (again, this is rather inefficient, and could probably be compressed into a single run if we wanted to do this again).

# In[]:


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


# # Pickle the data
#
# Finally, pickle the data for use in the remainder of the HTTPOT work.

# In[]:


all_data_features.drop(columns='ox_struct').to_pickle('full_featurized_data.pkl')

