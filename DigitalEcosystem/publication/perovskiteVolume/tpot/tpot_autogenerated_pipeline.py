import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectFwe, VarianceThreshold, f_regression
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: -3.695181140206585
exported_pipeline = make_pipeline(
    SelectFwe(score_func=f_regression, alpha=0.047),
    VarianceThreshold(threshold=0.2),
    SelectFwe(score_func=f_regression, alpha=0.046),
    StackingEstimator(estimator=RandomForestRegressor(bootstrap=True, max_features=0.6000000000000001, min_samples_leaf=16, min_samples_split=10, n_estimators=100)),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.2, min_samples_leaf=14, min_samples_split=14, n_estimators=100)),
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.01, max_depth=1, min_child_weight=8, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.7500000000000001, verbosity=0)),
    LassoLarsCV(normalize=True)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
