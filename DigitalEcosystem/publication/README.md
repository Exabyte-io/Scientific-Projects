This directory contains notebooks to reproduce the results and figures published in
"Interpretable Machine Learning for Materials Design" by Dean et al. The following notebooks are included in this
directory.

1. `featurize_all` - This notebook performs the structural featurization of the dataset.
2. `metalClassifier` - This notebook trains an XGBoost classifier to predict whether a 2D material is metallic
or not. On the data predicted to be metallic, we then train a regressor to predict the bandgap of the material.


Todo: Need these

1. `Perovskite Predictions` - This notebook reproduces the work surrounding the perovskite volume predictions problem.
2. `2D Material Bandgaps` - This notebook reproduces the work regarding 2D Material Bandgap prediction with SISSO,
Roost, XGBoost, and TPOT.
3. `2D Material Exfoliation Energy` - This notebook reproduces the work regarding the 2D Material Exfoliation Energy
prediction with SISSO, Roost, XGBoost, and TPOT
