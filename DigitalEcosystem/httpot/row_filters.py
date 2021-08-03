#!/usr/bin/env python
import pandas as pd


def rows_unfiltered(df: pd.DataFrame) -> pd.Series:
    """
    Always returns true

    Args:
        df (pd.DataFrame): A pandas dataframe
    """
    mask = df[df.columns[0]].apply(lambda i: True)
    return mask


def rows_reasonable(df: pd.DataFrame) -> pd.Series:
    """
    Returns true for elements that satisfy the following conditions:
        1) Not in the f-block
        2) Fewer than 92 protons
        3) Decomposition energy is below 0.5 eV

    Args:
        df (pd.DataFrame): A pandas dataframe
    """
    ignored_elements = ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
                        "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
                        "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Tg", "Cn", "Nh", "Fl", "Mc", "Mc", "Lv", "Ts", "Og",
                        "He", "Ne", "Ar", "Kr", "Xe", "Rn"]
    ignored_mask = df["atoms_object (unitless)"].apply(
        lambda atoms: all(symbol not in ignored_elements for symbol in atoms.get_chemical_symbols())
    )
    decomp_mask = df['decomposition_energy (eV/atom)'] <= 0.5
    mask = ignored_mask & decomp_mask
    return mask


def rows_bg_gte_100_mev(df: pd.DataFrame) -> pd.Series:
    """
    Returns true if the bandgap is greater than or equal to 0.1 eV, and all the conditions in
    defined in `rows_reasonable` are satisfied.

    Args:
        df (pd.DataFrame): A pandas dataframe
    """
    mask = rows_reasonable(df) & df['bandgap (eV)'] >= 0.1
    return mask
