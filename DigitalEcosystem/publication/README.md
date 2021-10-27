This directory contains notebooks to reproduce the results and figures published in
"Interpretable Machine Learning for Materials Design" by Dean et al. Each sub-directory contains one notebook, since
they all generate several files.

1. `featurize_all` - This notebook performs the structural featurization of the dataset.
2. `metalClassifier` - This notebook trains an XGBoost classifier to predict whether a 2D material is metallic
or not. On the data predicted to be metallic, we then train a regressor to predict the bandgap of the material.
3. `perovskiteVolume` - This notebook trains an XGBoost, TPOT, Roost, and SISSO model for the prediction of perovskite
   volumes.
4. `bandgap2DMaterials` - This notebook trains an XGBoost, TPOT, Roost, and SISSO model for the prediction of 2D
   Material bandgaps.
5. `exfoliation2DMaterials` - This notebook trains an XGBoost, TPOT, Roost, and SISSO model for the prediction of 2D
   Material bandgaps.

