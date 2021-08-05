#!/usr/bin/env python
# coding: utf-8

# In[]:


import pandas as pd
import sklearn
import tpot
import matplotlib.pyplot as plt


# In[]:


tmp_150_A = pd.read_csv("raw_data/tmp/Table_150_A.ssv", sep="\s+").set_index("Sr.No.")
tmp_150_B = pd.read_csv("raw_data/tmp/Table_150_B.ssv", sep="\s+").set_index("Sr.No.")
tmp_170_A = pd.read_csv("raw_data/tmp/Table_170_A.ssv", sep="\s+").set_index("Sr.No.")
tmp_170_B = pd.read_csv("raw_data/tmp/Table_170_B.ssv", sep="\s+").set_index("Sr.No.")

A = pd.concat([tmp_150_A, tmp_170_A])
B = pd.concat([tmp_150_B, tmp_170_B])


# In[]:


df = A.join(B.drop(columns="MXene")).drop(columns="BG_GW")
target="BG_PBE"

# Train / Test Split
data_train, data_test = sklearn.model_selection.train_test_split(df.drop(columns=["MXene"]), test_size=0.8, random_state=1234)
train_x = data_train.drop(columns=target).to_numpy()
train_y = data_train[target].to_numpy()

test_x = data_test.drop(columns=target).to_numpy()
test_y = data_test[target].to_numpy()

# Train a model
model = tpot.TPOTRegressor(
    generations=2,
    population_size=100,
    max_eval_time_mins=5/60,
    cv=3,
    verbosity=2,
    scoring="r2",
    config_dict=tpot.config.regressor_config_dict,
    n_jobs=6,
    random_state=1234
)


# In[]:


model.fit(train_x, train_y)


# In[]:


plt.rcParams["figure.figsize"]=[10,10]
train_y_pred = model.predict(train_x)
test_y_pred = model.predict(test_x)

plt.scatter(x=train_y_pred, y=train_y, label="Train Set")
plt.scatter(x=test_y_pred, y=test_y, label="Test Set")
min_val = min(map(min, train_y, test_y))
max_val = max(map(max, train_y, test_y))
plt.plot([min_val,max_val], [min_val,max_val], label="Parity")
plt.ylabel("Bandgap (Actual)")
plt.xlabel("Bandgap (Predicted)")
plt.legend()


# In[]:


data_train.columns.drop('BG_PBE')[model.fitted_pipeline_[0].get_support()]


# In[]:


model.fitted_pipeline_


# In[ ]:




