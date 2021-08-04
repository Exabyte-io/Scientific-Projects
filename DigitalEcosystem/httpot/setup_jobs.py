#!/usr/bin/env python
import functools
import random
import os
import datetime

import numpy as np
import pandas as pd

import sklearn.cluster
import sklearn.pipeline
import sklearn.decomposition

import fingerprints
import row_filters

RANDOM_SEED = 1234
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

pd.options.mode.chained_assignment = None

# ------------
# Read Dataset
# ------------
# Data with all features
data_path = "full_featurized_data.pkl"
data = pd.read_pickle(data_path)

# -------------
# Row Selection
# -------------
row_splits = {
    'all_rows': row_filters.rows_unfiltered,
    'reasonable_rows': row_filters.rows_reasonable,
    'reasonable_nonmetals_rows': row_filters.rows_bg_gte_100_mev
}

# ----------------
# Column Selection
# ----------------
xenonpy_descriptors = [col for col in data.columns if ":" in col]
matminer_descriptors = [
    'bond_length_average',
    'bond_angle_average',
    'average_cn',
    'global_instability',
    'perimeter_area_ratio',
    'ewald_energy_per_atom',
    'structural complexity per atom',
    'structural complexity per cell',
    'n_symmetry_ops'

]
xenonpy_matminer_descriptors = xenonpy_descriptors + matminer_descriptors
length_angle_features = [col for col in data.columns if
                         any([unit in col for unit in ['(atoms)', '(bonds)', '(angles)']])]
column_splits = {
    'lengthAngle_cols': length_angle_features,
    'xenonpy_matminer_cols': xenonpy_matminer_descriptors,
    'all_cols': xenonpy_matminer_descriptors + length_angle_features
}

# Also, need to fillna on these
data[length_angle_features] = data[length_angle_features].fillna(0)

# -------------------
# Fingerprint Methods
# -------------------
cols_to_drop = ['formula',
                'discovery_process (unitless)',
                'potcars (unitless)',
                'is_hubbard (unitless)',
                'energy_per_atom (eV)',
                'exfoliation_energy_per_atom (eV/atom)',
                'is_bandgap_direct (unitless)',
                'is_metal (unitless)',
                'energy_vdw_per_atom (eV/atom)',
                'total_magnetization (Bohr Magneton)']
target_column = ['bandgap (eV)']
matpedia_id = ['2dm_id (unitless)']
atoms_col = ['atoms_object (unitless)']

remove_after_row_selection = ['decomposition_energy (eV/atom)']

max_atoms = max(data['atoms_object (unitless)'].apply(len))

cached_fingerprints = ['soap_fingerprint', 'ewald_fingerprint']
fingerprints = {
    'ewald_fingerprint': {
        "fun": functools.partial(fingerprints.fingerprint_ewald_sum,
                                 max_atoms),
        "print_train": None,
        "print_test": None
    },
    'direct_fingerprint': fingerprints.fingerprint_df,

    'soap_fingerprint': {
        "fun": fingerprints.fingerprint_soap,
        "print_train": None,
        "print_test": None
    },
}


def get_best_k(fingerprint: np.ndarray) -> int:
    """
    Given a fingerprint, optimizes the k-value for k-means.
    Uses minibatch k-means, since there can be quite a few structures to look at

    """
    best_k = 2
    hiscore = -np.inf
    best_kmeans = None

    n_pca_dims = min(fingerprint.shape[1], 128)
    pca = sklearn.decomposition.PCA(n_components=n_pca_dims)
    pca_cache = pca.fit_transform(fingerprint)
    for k in range(2, 6):
        kmeans = sklearn.cluster.KMeans(n_clusters=k, random_state=RANDOM_SEED)
        kmeans.fit(pca_cache)
        score = sklearn.metrics.calinski_harabasz_score(fingerprint, kmeans.predict(pca_cache))
        if score > hiscore:
            best_k = k
            hiscore = score

        print(f"         |---Trying K={k} and PCA_dims={n_pca_dims}, score={np.round(score, 3)}, hiscore={np.round(hiscore, 3)} at {best_k}",
              flush=True)
    best_kmeans = sklearn.pipeline.make_pipeline(
        sklearn.decomposition.PCA(n_components=n_pca_dims),
        sklearn.cluster.KMeans(n_clusters=best_k, random_state=RANDOM_SEED)
    )
    best_kmeans.fit(fingerprint)
    return best_k, best_kmeans


def write_calculation(row_pathname: str,
                      col_pathname: str,
                      finger_pathname: str,
                      model_pathname: str,
                      best_k,
                      train: pd.DataFrame,
                      test: pd.DataFrame) -> None:
    csv_data = {
        "filter": row_pathname,
        "features": col_pathname,
        "fingerprint": finger_pathname,
        "model": model_pathname,
        "K": "",
        "status": "Pending",
        "setup_date_utc": "",
        "run_date_utc": "NA",
        "MAE": "NA",
        "MSE": "NA",
        "RMSE": "NA",
        "MAPE": "NA",
        "R2": "NA",
        "MaxError": "NA"
    }

    basename = "httpot_runs"
    model_path = os.path.join(".", basename, row_pathname, col_pathname, finger_pathname, model_pathname)
    # First off, generate the runs for the individual K's
    for k in range(best_k):
        path = os.path.join(model_path, f"k_{k}")
        os.makedirs(path, exist_ok=True)
        print(f"             |---Writing to {path}")
        train[train["k_means_label"] == k].drop(columns=["k_means_label"]).to_parquet(
            path=os.path.join(path, "train.parquet"),
            engine="pyarrow",
            compression="gzip")
        test[test["k_means_label"] == k].drop(columns=["k_means_label"]).to_parquet(
            path=os.path.join(path, "test.parquet"),
            engine="pyarrow",
            compression="gzip")
        with open(os.path.join(path, "result.csv"), "w") as outp:
            csv_data["setup_date_utc"] = datetime.datetime.utcnow().isoformat()
            csv_data["K"] = str(k)
            outp.write(",".join(csv_data.keys()) + "\n")
            outp.write(",".join(csv_data.values()) + "\n")

    # Finally, generate the all-k run
    path = os.path.join(model_path, "k_as_column")
    os.makedirs(path, exist_ok=True)
    print(f"             |---Writing to {path}")
    train.to_parquet(path=os.path.join(path, "train.parquet"),
                     engine="pyarrow",
                     compression="gzip")
    test.to_parquet(path=os.path.join(path, "test.parquet"),
                    engine="pyarrow",
                    compression="gzip")
    with open(os.path.join(path, "result.csv"), "w") as outp:
        csv_data["setup_date_utc"] = datetime.datetime.utcnow().isoformat()
        csv_data["K"] = "k_as_column"
        outp.write(",".join(csv_data.keys()) + "\n")
        outp.write(",".join(csv_data.values()) + "\n")


data_train, data_test = sklearn.model_selection.train_test_split(data,
                                                                 test_size=0.1,
                                                                 random_state=RANDOM_SEED)

for col_pathname, col_selection in column_splits.items():
    print(f"Cols are {col_pathname}")
    for finger_pathname, finger in fingerprints.items():
        # SOAP and Ewald Fingerprints
        if finger_pathname in cached_fingerprints:
            if finger['print_train'] is None:
                print(f"|---Caching {finger_pathname}")
                finger['print_train'] = finger['fun'](data_train)
                finger['print_test'] = finger['fun'](data_test)
            print(f"|---Recalled {finger_pathname} from cache")
            print_train = np.nan_to_num(finger['print_train'])
            print_test = np.nan_to_num(finger['print_test'])

        # Direct Fingerprints
        else:
            print_train = np.nan_to_num(finger(data_train[col_selection]))
            print_test = np.nan_to_num(finger(data_test[col_selection]))
            print(f"|---Calculated {finger_pathname}, cannot be cached")

        for row_pathname, row_selection in row_splits.items():
            # Filter rows
            print(f"    |---{row_pathname}")
            filtered_train = data_train.copy()
            train_mask = row_selection(df=filtered_train)
            filtered_train = filtered_train[train_mask]

            filtered_test = data_test.copy()
            test_mask = row_selection(df=filtered_test)
            filtered_test = filtered_test[test_mask]

            # Filter fingerprints
            filtered_train_prints = print_train[train_mask]
            filtered_test_prints = print_test[test_mask]

            # Find best k
            best_k, best_kmeans = get_best_k(filtered_train_prints)
            filtered_train["k_means_label"] = best_kmeans.predict(filtered_train_prints)
            filtered_test["k_means_label"] = best_kmeans.predict(filtered_test_prints)

            write_calculation(row_pathname=row_pathname,
                              col_pathname=col_pathname,
                              finger_pathname=finger_pathname,
                              model_pathname='tpot',
                              best_k=best_k,
                              train=filtered_train.drop(columns=cols_to_drop + remove_after_row_selection + atoms_col),
                              test=filtered_test.drop(columns=cols_to_drop + remove_after_row_selection + atoms_col))
