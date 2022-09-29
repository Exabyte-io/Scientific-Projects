import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFwe, f_regression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator, ZeroCount
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: -0.3975520720416017
exported_pipeline = make_pipeline(
    StandardScaler(),
    StackingEstimator(estimator=RidgeCV()),
    ZeroCount(),
    MaxAbsScaler(),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.7000000000000001, min_samples_leaf=6, min_samples_split=15, n_estimators=100)),
    SelectFwe(score_func=f_regression, alpha=0.011),
    StackingEstimator(estimator=LinearSVR(C=0.5, dual=True, epsilon=0.0001, loss="squared_epsilon_insensitive", tol=1e-05)),
    MinMaxScaler(),
    ExtraTreesRegressor(bootstrap=False, max_features=0.1, min_samples_leaf=1, min_samples_split=5, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
