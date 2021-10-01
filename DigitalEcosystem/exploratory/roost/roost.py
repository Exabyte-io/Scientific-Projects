#!/usr/bin/env python
# coding: utf-8

# In[]:


import matplotlib.pyplot as plt
import pandas as pd


# In[]:


bandgap_train = pd.read_csv('bandgap_preds_train.csv')
bandgap_test = pd.read_csv('bandgap_preds_test.csv')
exfoliation_train = pd.read_csv('exfol_preds_train.csv')
exfoliation_test = pd.read_csv('exfol_preds_test.csv')


# In[]:


bandgap_train


# In[]:


plt.rcParams['figure.figsize'] = [10,10]
plt.rcParams['font.size'] = 16
plt.scatter(x=bandgap_train['bandgap_target'], y=bandgap_train['bandgap_pred_n0'], label='Training Set')
plt.scatter(x=bandgap_test['bandgap_target'], y=bandgap_test['bandgap_pred_n0'], label='Testing Set')
plt.plot([0,9],[0,9], c='k', label='Parity')
plt.xlim([0,9])
plt.ylim([0,9])
plt.legend()
plt.xlabel('DFT Bandgap (eV)')
plt.ylabel('Predicted Bandgap (eV)')


# In[]:


plt.rcParams['figure.figsize'] = [10,10]
plt.rcParams['font.size'] = 16
plt.scatter(x=exfoliation_train['exfoliation_energy_target'], y=exfoliation_train['exfoliation_energy_pred_n0'], label='Training Set')
plt.scatter(x=exfoliation_test['exfoliation_energy_target'], y=exfoliation_test['exfoliation_energy_pred_n0'], label='Testing Set')
plt.plot([0,10],[0,10], c='k', label='Parity')
plt.xlim([0,10])
plt.ylim([0,10])
plt.legend()
plt.xlabel('DFT Exfoliation Energy (eV)')
plt.ylabel('Predicted Exfoliation Energy (eV)')


# In[]:


exfoliation_train


# In[ ]:




