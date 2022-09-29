import functools
import pickle
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tpot
import sklearn
import optuna
import ase.io
import xgboost
import pymatgen
# import xenonpy.descriptor
from tqdm.notebook import tqdm 
import sys, os

sys.path.append("../../../")
import DigitalEcosystem.utils.figures
from DigitalEcosystem.utils.functional import except_with_default_value
from DigitalEcosystem.utils.misc import matminer_descriptors, root_mean_squared_error
from DigitalEcosystem.utils.element_symbols import noble_gases, f_block_elements, synthetic_elements_in_d_block

from IPython.display import Latex

pd.options.mode.chained_assignment = None 
tqdm.pandas()

# Random seeds for reproducibility
RANDOM_SEED = 42
import random
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Plot Configuration
plt.rcParams["figure.figsize"] = (15, 15)
plt.rcParams["font.size"] = 32

data = pd.read_csv('perovskite_data.csv', index_col=0)

target_column = ['Volume']
descriptors = [
    "ave:atomic_number",
    "ave:atomic_radius",
    "ave:atomic_radius_rahm",
    "ave:atomic_volume",
    "ave:atomic_weight",
    "ave:boiling_point",
    "ave:bulk_modulus",
    "ave:c6_gb",
    "ave:covalent_radius_cordero",
    "ave:covalent_radius_pyykko",
    "ave:covalent_radius_pyykko_double",
    "ave:covalent_radius_pyykko_triple",
    "ave:covalent_radius_slater",
    "ave:density",
    "ave:dipole_polarizability",
    "ave:electron_negativity",
    "ave:electron_affinity",
    "ave:en_allen",
    "ave:en_ghosh",
    "ave:en_pauling",
    "ave:first_ion_en",
    "ave:fusion_enthalpy",
    "ave:gs_bandgap",
    "ave:gs_energy",
    "ave:gs_est_bcc_latcnt",
    "ave:gs_est_fcc_latcnt",
    "ave:gs_mag_moment",
    "ave:gs_volume_per",
    "ave:hhi_p",
    "ave:hhi_r",
    "ave:heat_capacity_mass",
    "ave:heat_capacity_molar",
    "ave:icsd_volume",
    "ave:evaporation_heat",
    "ave:heat_of_formation",
    "ave:lattice_constant",
    "ave:mendeleev_number",
    "ave:melting_point",
    "ave:molar_volume",
    "ave:num_unfilled",
    "ave:num_valance",
    "ave:num_d_unfilled",
    "ave:num_d_valence",
    "ave:num_f_unfilled",
    "ave:num_f_valence",
    "ave:num_p_unfilled",
    "ave:num_p_valence",
    "ave:num_s_unfilled",
    "ave:num_s_valence",
    "ave:period",
    "ave:specific_heat",
    "ave:thermal_conductivity",
    "ave:vdw_radius",
    "ave:vdw_radius_alvarez",
    "ave:vdw_radius_mm3",
    "ave:vdw_radius_uff",
    "ave:sound_velocity",
    "ave:Polarizability",
    "sum:atomic_number",
    "sum:atomic_radius",
    "sum:atomic_radius_rahm",
    "sum:atomic_volume",
    "sum:atomic_weight",
    "sum:boiling_point",
    "sum:bulk_modulus",
    "sum:c6_gb",
    "sum:covalent_radius_cordero",
    "sum:covalent_radius_pyykko",
    "sum:covalent_radius_pyykko_double",
    "sum:covalent_radius_pyykko_triple",
    "sum:covalent_radius_slater",
    "sum:density",
    "sum:dipole_polarizability",
    "sum:electron_negativity",
    "sum:electron_affinity",
    "sum:en_allen",
    "sum:en_ghosh",
    "sum:en_pauling",
    "sum:first_ion_en",
    "sum:fusion_enthalpy",
    "sum:gs_bandgap",
    "sum:gs_energy",
    "sum:gs_est_bcc_latcnt",
    "sum:gs_est_fcc_latcnt",
    "sum:gs_mag_moment",
    "sum:gs_volume_per",
    "sum:hhi_p",
    "sum:hhi_r",
    "sum:heat_capacity_mass",
    "sum:heat_capacity_molar",
    "sum:icsd_volume",
    "sum:evaporation_heat",
    "sum:heat_of_formation",
    "sum:lattice_constant",
    "sum:mendeleev_number",
    "sum:melting_point",
    "sum:molar_volume",
    "sum:num_unfilled",
    "sum:num_valance",
    "sum:num_d_unfilled",
    "sum:num_d_valence",
    "sum:num_f_unfilled",
    "sum:num_f_valence",
    "sum:num_p_unfilled",
    "sum:num_p_valence",
    "sum:num_s_unfilled",
    "sum:num_s_valence",
    "sum:period",
    "sum:specific_heat",
    "sum:thermal_conductivity",
    "sum:vdw_radius",
    "sum:vdw_radius_alvarez",
    "sum:vdw_radius_mm3",
    "sum:vdw_radius_uff",
    "sum:sound_velocity",
    "sum:Polarizability",
    "var:atomic_number",
    "var:atomic_radius",
    "var:atomic_radius_rahm",
    "var:atomic_volume",
    "var:atomic_weight",
    "var:boiling_point",
    "var:bulk_modulus",
    "var:c6_gb",
    "var:covalent_radius_cordero",
    "var:covalent_radius_pyykko",
    "var:covalent_radius_pyykko_double",
    "var:covalent_radius_pyykko_triple",
    "var:covalent_radius_slater",
    "var:density",
    "var:dipole_polarizability",
    "var:electron_negativity",
    "var:electron_affinity",
    "var:en_allen",
    "var:en_ghosh",
    "var:en_pauling",
    "var:first_ion_en",
    "var:fusion_enthalpy",
    "var:gs_bandgap",
    "var:gs_energy",
    "var:gs_est_bcc_latcnt",
    "var:gs_est_fcc_latcnt",
    "var:gs_mag_moment",
    "var:gs_volume_per",
    "var:hhi_p",
    "var:hhi_r",
    "var:heat_capacity_mass",
    "var:heat_capacity_molar",
    "var:icsd_volume",
    "var:evaporation_heat",
    "var:heat_of_formation",
    "var:lattice_constant",
    "var:mendeleev_number",
    "var:melting_point",
    "var:molar_volume",
    "var:num_unfilled",
    "var:num_valance",
    "var:num_d_unfilled",
    "var:num_d_valence",
    "var:num_f_unfilled",
    "var:num_f_valence",
    "var:num_p_unfilled",
    "var:num_p_valence",
    "var:num_s_unfilled",
    "var:num_s_valence",
    "var:period",
    "var:specific_heat",
    "var:thermal_conductivity",
    "var:vdw_radius",
    "var:vdw_radius_alvarez",
    "var:vdw_radius_mm3",
    "var:vdw_radius_uff",
    "var:sound_velocity",
    "var:Polarizability",
    "max:atomic_number",
    "max:atomic_radius",
    "max:atomic_radius_rahm",
    "max:atomic_volume",
    "max:atomic_weight",
    "max:boiling_point",
    "max:bulk_modulus",
    "max:c6_gb",
    "max:covalent_radius_cordero",
    "max:covalent_radius_pyykko",
    "max:covalent_radius_pyykko_double",
    "max:covalent_radius_pyykko_triple",
    "max:covalent_radius_slater",
    "max:density",
    "max:dipole_polarizability",
    "max:electron_negativity",
    "max:electron_affinity",
    "max:en_allen",
    "max:en_ghosh",
    "max:en_pauling",
    "max:first_ion_en",
    "max:fusion_enthalpy",
    "max:gs_bandgap",
    "max:gs_energy",
    "max:gs_est_bcc_latcnt",
    "max:gs_est_fcc_latcnt",
    "max:gs_mag_moment",
    "max:gs_volume_per",
    "max:hhi_p",
    "max:hhi_r",
    "max:heat_capacity_mass",
    "max:heat_capacity_molar",
    "max:icsd_volume",
    "max:evaporation_heat",
    "max:heat_of_formation",
    "max:lattice_constant",
    "max:mendeleev_number",
    "max:melting_point",
    "max:molar_volume",
    "max:num_unfilled",
    "max:num_valance",
    "max:num_d_unfilled",
    "max:num_d_valence",
    "max:num_f_unfilled",
    "max:num_f_valence",
    "max:num_p_unfilled",
    "max:num_p_valence",
    "max:num_s_unfilled",
    "max:num_s_valence",
    "max:period",
    "max:specific_heat",
    "max:thermal_conductivity",
    "max:vdw_radius",
    "max:vdw_radius_alvarez",
    "max:vdw_radius_mm3",
    "max:vdw_radius_uff",
    "max:sound_velocity",
    "max:Polarizability",
    "min:atomic_number",
    "min:atomic_radius",
    "min:atomic_radius_rahm",
    "min:atomic_volume",
    "min:atomic_weight",
    "min:boiling_point",
    "min:bulk_modulus",
    "min:c6_gb",
    "min:covalent_radius_cordero",
    "min:covalent_radius_pyykko",
    "min:covalent_radius_pyykko_double",
    "min:covalent_radius_pyykko_triple",
    "min:covalent_radius_slater",
    "min:density",
    "min:dipole_polarizability",
    "min:electron_negativity",
    "min:electron_affinity",
    "min:en_allen",
    "min:en_ghosh",
    "min:en_pauling",
    "min:first_ion_en",
    "min:fusion_enthalpy",
    "min:gs_bandgap",
    "min:gs_energy",
    "min:gs_est_bcc_latcnt",
    "min:gs_est_fcc_latcnt",
    "min:gs_mag_moment",
    "min:gs_volume_per",
    "min:hhi_p",
    "min:hhi_r",
    "min:heat_capacity_mass",
    "min:heat_capacity_molar",
    "min:icsd_volume",
    "min:evaporation_heat",
    "min:heat_of_formation",
    "min:lattice_constant",
    "min:mendeleev_number",
    "min:melting_point",
    "min:molar_volume",
    "min:num_unfilled",
    "min:num_valance",
    "min:num_d_unfilled",
    "min:num_d_valence",
    "min:num_f_unfilled",
    "min:num_f_valence",
    "min:num_p_unfilled",
    "min:num_p_valence",
    "min:num_s_unfilled",
    "min:num_s_valence",
    "min:period",
    "min:specific_heat",
    "min:thermal_conductivity",
    "min:vdw_radius",
    "min:vdw_radius_alvarez",
    "min:vdw_radius_mm3",
    "min:vdw_radius_uff",
    "min:sound_velocity",
    "min:Polarizability",
]
train, test = sklearn.model_selection.train_test_split(data, test_size=0.1, random_state=RANDOM_SEED)

train_x = np.nan_to_num(train[descriptors].to_numpy())
train_y = np.nan_to_num(train[target_column].to_numpy())

test_x = np.nan_to_num(test[descriptors].to_numpy())
test_y = np.nan_to_num(test[target_column].to_numpy())

def rmse(y_true, y_pred):
    mse = sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)
    rmse = np.sqrt(abs(mse))
    return rmse

metrics = {
    'MaxError': sklearn.metrics.max_error,
    'MAE': sklearn.metrics.mean_absolute_error,
    'MSE': sklearn.metrics.mean_squared_error,
    'RMSE': root_mean_squared_error,
    'MAPE': sklearn.metrics.mean_absolute_percentage_error,
    'R2': sklearn.metrics.r2_score
}


tpot_model = tpot.TPOTRegressor(
    generations=100,
    population_size=100,
#     max_eval_time_mins=10 / 60,
    cv=10,
    verbosity=2,
    scoring="neg_root_mean_squared_error",
    config_dict=tpot.config.regressor_config_dict,
    n_jobs=-1,
    random_state=RANDOM_SEED
)

tpot_model.fit(train_x, train_y.ravel())
try:
    tpot_importances = zip(tpot_model.fitted_pipeline_[1].feature_importances_, descriptors)
    sorted_tpot_importances = list(sorted(tpot_importances, key=lambda i: -abs(i[0])))

    old_figsize = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = (2*old_figsize[0], old_figsize[1])

    plt.barh(range(n_importances), [imp[0] for imp in sorted_tpot_importances[:n_importances]])
    plt.yticks(range(n_importances), [imp[1] for imp in sorted_tpot_importances[:n_importances]])
    plt.ylabel("Feature")
    plt.xlabel("Extra Trees Importance")
    plt.tight_layout()
    plt.show()
    plt.savefig("tpot_perovskite_volume_importances.jpeg")
    plt.close()

    plt.rcParams["figure.figsize"] = old_figsize
except:
	old_figsize = plt.rcParams["figure.figsize"]

plt.rcParams['figure.figsize'] = old_figsize
train_preds = train[target_column]
train_preds['TrainTest Status'] = ['Training Set'] * len(train_preds)
train_preds['Prediction'] = tpot_model.predict(train_x)

test_preds = test[target_column]
test_preds['TrainTest Status'] = ['Test Set'] * len(test_preds)
test_preds['Prediction'] = tpot_model.predict(test_x)

DigitalEcosystem.utils.figures.save_parity_plot_publication_quality(train_y_true = train_y,
                                                                    train_y_pred = tpot_model.predict(train_x),
                                                                    test_y_true = test_y,
                                                                    test_y_pred = tpot_model.predict(test_x),
                                                                    axis_label = "Perovskite Volume (Å^3 / formula unit)",
                                                                    filename = "tpot_perovskite_volume_parity.jpeg")
print("Test Set Error Metrics")
for key, fun in metrics.items():
    value = fun(y_true=test_y, y_pred=tpot_model.predict(test_x))
    print(key,np.round(value,4))
    
print("\nTraining Set Error Metrics")
for key, fun in metrics.items():
    value = fun(y_true=train_y, y_pred=tpot_model.predict(train_x))
    print(key,np.round(value,4))

train_preds = train[target_column]
train_preds['TrainTest Status'] = ['Training Set'] * len(train_preds)
train_preds['Prediction'] = tpot_model.predict(train_x)

test_preds = test[target_column]
test_preds['TrainTest Status'] = ['Test Set'] * len(test_preds)
test_preds['Prediction'] = tpot_model.predict(test_x)

tpot_predictions = train_preds.append(test_preds)
tpot_predictions.to_csv("tpot_perovskite_predictions.csv")

tpot_model.export('tpot_autogenerated_pipeline.py')
with open("tpot_pipeline.pkl", "wb") as outp:
    pickle.dump(tpot_model.fitted_pipeline_, outp)



with open("tpot_pipeline.pkl", "wb") as outp:
    pickle.dump(tpot_model.fitted_pipeline_, outp)

