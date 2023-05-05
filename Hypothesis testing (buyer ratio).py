#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency


# In[31]:


df=pd.read_csv('BuyerRatio.csv')
df


# In[32]:


x1=np.array([[50,142,131,70],[435,1523,1356,750]])
x1


# In[33]:


chi2_contingency(x1)
#  o/p is (Chi2 stats value, p_value, df, expected obsvations)


# In[37]:


tabvalue = stats.chi2.ppf(q=0.95,df=3)


# In[38]:


pvalue =0.66
alpha = 0.05


# In[36]:


# h0 = All proportions are equal.
# ha = Not all Proportions are equal.
if pvalue < alpha :
    print ("h0 is accepted and ha is rejected ")
else:
    print ("ha is accepted and h0 is rejected")


# In[ ]:




