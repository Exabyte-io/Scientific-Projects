import ase
import pandas as pd
import numpy as np
import dscribe.descriptors
from .functional import except_with_default_value

import functools

from matminer.featurizers.site import AverageBondLength, AverageBondAngle
from matminer.featurizers.structure import GlobalInstabilityIndex
from pymatgen.analysis.local_env import JmolNN


# ====================================================================================================================
# Fingerprints
# These are properties which do not have a direct interpretation, and provide a unique (or near-unique) representation
# of the structure.
# ====================================================================================================================

def fingerprint_df(df: pd.DataFrame) -> np.ndarray:
    """
    Converts the regression-relevant parts of a pandas dataframe into a numpy array

    Args:
        df (pd.DataFrame): A pandas dataframe containing the data

    Returns:
        A numpy array containing the regression-relevant parts of the data
    """
    result = df.to_numpy()
    return result


def fingerprint_soap(df: pd.DataFrame) -> np.ndarray:
    """
    Generates the Smooth Overlap of Atomic Positions (SOAP) fingerprint using Dscribe:
    https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html

    In systems where multiple unique atoms exist, concatenates the mean/min/max of the atomic soaps into a single
    array, intended for use in clustering. PCA it beforehand, there's a *lot* of dimensions.

    Args:
        df (pd.DataFrame): A pandas dataframe containing the data

    Returns:
        A fairly large numpy array.
    """
    symbols = range(1, 95)

    soap = dscribe.descriptors.SOAP(species=symbols,
                                    periodic=True,
                                    rcut=4,
                                    nmax=2,
                                    lmax=4)

    def inner(atoms: ase.Atoms):
        lathered = soap.create(atoms, n_jobs=4)
        # Soap creates an N x M array
        #     - N is the number of atoms in the system
        #     - M is the size of the SOAP descriptor
        # So we'll average along the N direction
        rinsed = np.hstack([lathered.mean(axis=0), lathered.min(axis=0), lathered.max(axis=0)])
        return rinsed

    result = np.real(np.vstack(df['atoms_object (unitless)'].apply(inner)))
    return result


def fingerprint_ewald_sum(max_atoms: int, df: pd.DataFrame, ) -> np.ndarray:
    """
    Takes the eigenspectrum of the Ewald sum using Dscribe:
    https://singroup.github.io/dscribe/latest/tutorials/descriptors/ewald_sum_matrix.html

    Args:
        df (pd.DataFrame): A pandas dataframe containing the data

    Returns:
        An array of dimension max_atoms
    """
    ewald_matrix = dscribe.descriptors.EwaldSumMatrix(n_atoms_max=max_atoms,
                                                      permutation='eigenspectrum',
                                                      sparse=False)

    result = np.real(np.vstack(df['atoms_object (unitless)'].apply(ewald_matrix.create)))

    return result


# =============================================================================================================================
# Descriptors
# These are properties which typically do have a direct interpretation, and are not intended to provide a unique representation
# of the structure.
# =============================================================================================================================

@except_with_default_value(exceptions_to_catch=(BaseException,),
                           default_return=None)
def global_instability(struct):
    return desc.featurize(struct)[0]


neighbor_finder = JmolNN()

default_val_on_index_error = functools.partial(except_with_default_value,
                                               exceptions_to_catch=(IndexError,),
                                               default_return=None)


@default_val_on_index_error()
def average_bond_length(structure, featurizer=AverageBondLength(neighbor_finder)):
    n_atoms = len(structure)
    lengths = map(lambda i: featurizer.featurize(structure, i)[0], range(n_atoms))
    return sum(lengths) / n_atoms


@default_val_on_index_error()
def average_bond_angle(structure, featurizer=AverageBondAngle(neighbor_finder)):
    n_atoms = len(structure)
    angles = map(lambda i: featurizer.featurize(structure, i)[0], range(n_atoms))
    return sum(angles) / n_atoms


def average_cn(structure, neighbor_finder=neighbor_finder):
    n_atoms = len(structure)
    cns = map(lambda i: neighbor_finder.get_cn(structure, i), range(n_atoms))
    return sum(cns) / n_atoms


def ab_perimeter_area_ratio(structure):
    a, b, c = structure.lattice.matrix
    perimeter = 2 * np.linalg.norm(a) + 2 * np.linalg.norm(b)
    area = np.linalg.norm(np.cross(a, b))
    return perimeter / area


desc = GlobalInstabilityIndex()
