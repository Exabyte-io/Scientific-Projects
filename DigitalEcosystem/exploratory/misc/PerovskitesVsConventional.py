#!/usr/bin/env python
# coding: utf-8

# In[]:


import collections
import ase.data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model

import sklearn.metrics

from xenonpy.descriptor import Compositions


# In[]:


dataset_path = './DigitalEcosystem/raw_data/perovskites.pkl'
df = pd.read_pickle(dataset_path)
df["Volume"] /= df["Atoms_Object"].apply(lambda atoms: len(atoms)//5)

# Featurize with XenonPy
cal = Compositions()
df["Symbols"] = df.Atoms_Object.apply(lambda atoms: collections.Counter(atoms.get_chemical_symbols()))
featurized_data = pd.concat([df, cal.transform(df.Symbols)], axis=1)
featurized_data


# In[]:


def mape(x):
    mape = sklearn.metrics.mean_absolute_percentage_error(y_pred=x, y_true=df['Volume'])
    return np.round(mape, 3)


# In[]:


sisso_preds = -172 + 2.63 * (featurized_data["min:atomic_volume"] + featurized_data["ave:covalent_radius_pyykko_triple"])

summed_covalent_radii = df["Atoms_Object"].apply(lambda atoms: sum(ase.data.covalent_radii[atoms.get_atomic_numbers()])).to_numpy().reshape(-1,1)
model = sklearn.linear_model.LinearRegression()

ave = featurized_data["ave:covalent_radius_pyykko_triple"].to_numpy().reshape(-1,1)
model.fit(ave, df["Volume"])
ols_preds_ave = model.predict(ave)

mins = featurized_data['min:atomic_volume'].to_numpy().reshape(-1,1)
model.fit(mins, df['Volume'])
ols_preds_min = model.predict(mins)

sums = featurized_data["sum:covalent_radius_pyykko_triple"].to_numpy().reshape(-1,1)
model.fit(sums, df["Volume"])
ols_preds_sum = model.predict(sums)

plt.rcParams['figure.figsize'] = [16,16]
plt.rcParams['font.size'] = 16

plt.scatter(x=df["Volume"], y=sisso_preds, label=f"SISSO (y=c + a0 * (min:atomic_volume + ave:covalent_radius_pyykko_triple), MAPE={mape(sisso_preds)}")
plt.scatter(x=df["Volume"], y=ols_preds_ave, label=f"OLS(ave:covalent_radius_pyykko_triple), MAPE={mape(ols_preds_ave)}")
#plt.scatter(x=df['Volume'], y=ols_preds_min, label=f"OLS(min:atomic_volume), MAPE={mape(ols_preds_min)}")

#plt.scatter(x=df["Volume"], y=ols_preds_sum, label=f"OLS(sum:covalent_radius_pyykko_triple), MAPE={mape(ols_preds_sum)}")

plt.plot([0,max(df["Volume"])], [0,max(df["Volume"])], label="Parity")
plt.ylabel("Volume (Predicted, Å^3 per Formula Unit)")
plt.xlabel("Volume (Actual, Å^3 per Formula Unit)")
plt.legend()


# In[ ]:




