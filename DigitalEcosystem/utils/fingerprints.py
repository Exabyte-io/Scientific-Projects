import ase
import pandas as pd
import numpy as np
import dscribe.descriptors

import functools


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


def fingerprint_ewald_sum(max_atoms: int, df: pd.DataFrame,) -> np.ndarray:
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
