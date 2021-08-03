import os
import datetime
import random
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.metrics
import tpot

from tpot_config import IMPORTANCE_CONFIG

RANDOM_SEED = 1234
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------
# Train the Model
# ---------------
# Go to the job

# Load the data
print("Loading the training set")
data_train = pd.read_parquet('train.parquet')
data_test = pd.read_parquet('test.parquet')

try:
    # Train the model
    model = tpot.TPOTRegressor(
        generations=None,
        population_size=100,
        max_eval_time_mins=1,
        max_time_mins=60,
        cv=5,
        verbosity=2,
        scoring="neg_mean_squared_error",
        config_dict=IMPORTANCE_CONFIG,
        n_jobs=-1,
        memory="auto",
        random_state=RANDOM_SEED
    )

    print("Fitting the model")
    model.fit(features=data_train.drop(columns=['2dm_id (unitless)', 'bandgap (eV)']),
              target=data_train['bandgap (eV)'])

    # Save the model to disk
    print("Exporting model script")
    model.export('optimized_model.py')

    # --------
    # Evaluate
    # --------

    # Evaluate the model
    print("Evaluating model accuracy")
    metrics = {
        'MaxError': sklearn.metrics.max_error,
        'MAE': sklearn.metrics.mean_absolute_error,
        'MSE': sklearn.metrics.mean_squared_error,
        'MAPE': sklearn.metrics.mean_absolute_percentage_error,
        'R2': sklearn.metrics.r2_score
    }

    y_true_train = data_train['bandgap (eV)'].to_numpy().reshape(-1, 1)
    y_true_test = data_test['bandgap (eV)'].to_numpy().reshape(-1, 1)

    y_pred_train = model.predict(data_train.drop(columns=['2dm_id (unitless)', 'bandgap (eV)']))
    y_pred_test = model.predict(data_test.drop(columns=['2dm_id (unitless)', 'bandgap (eV)']))

    # -----------------
    # Store the results
    # -----------------
    print("Storing model accuracy")
    results = pd.read_csv('result.csv')
    for key, fun in metrics.items():
        try:
            value = fun(y_true=y_true_test, y_pred=y_pred_test)
        except:
            value = "NA"
        results[key] = value

    # ----------------
    # Plot the results
    # ----------------
    print("Generating parity plot")
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
    plt.rcParams["figure.figsize"] = (15, 15)
    plt.rcParams["font.size"] = 16

    plt.scatter(x=y_true_train, y=y_pred_train, label="Train Set")
    plt.scatter(x=y_true_test, y=y_pred_test, label="Test Set")
    min_xy = min(min(y_true_test), min(y_true_train), min(y_pred_test), min(y_pred_train))
    max_xy = max(max(y_true_test), max(y_true_train), max(y_pred_test), max(y_pred_train))

    plt.plot([min_xy, max_xy], [min_xy, max_xy], label="Parity")
    plt.ylabel("Bandgap (Dataset)")
    plt.xlabel("Bandgap (Predicted)")
    plt.legend()
    plt.savefig('plot.jpeg')

    # ----------------------
    # Store some predictions
    # ----------------------
    print("Storing prediction results")
    pd.DataFrame.from_dict({
        '2dm_id (unitless)': pd.concat([data_train['2dm_id (unitless)'], data_test['2dm_id (unitless)']]),
        'Training_Set': [1] * len(data_train['2dm_id (unitless)']) + [0] * len(data_test['2dm_id (unitless)']),
        'bandgap (eV)': pd.concat([data_train['bandgap (eV)'], data_test['bandgap (eV)']]),
        'pred_bandgap (eV)': list(y_pred_train) + list(y_pred_test)
    }).to_csv('predictions.csv')

    # ----------------
    # Pickle the Model
    # ----------------
    print("Pickling the TPOT model")
    with open("best_pipeline.pkl", "wb") as outp:
        pickle.dump(model.fitted_pipeline_, outp)

except:
    print("Error encountered")
    results = pd.read_csv('result.csv')
    results['run_date_utc'] = datetime.datetime.utcnow().isoformat()
    results['status'] = 'Error'
    results.to_csv('result.csv', index=False)
    raise

print("Run successful, updating result.csv")
results['run_date_utc'] = datetime.datetime.utcnow().isoformat()
results['status'] = 'Complete'
results.to_csv('result.csv', index=False)

