#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.stats import chi2_contingency


# In[3]:


df= pd.read_csv('Costomer+OrderForm.csv')
df


# In[5]:


df.Phillippines.value_counts()


# In[6]:


df.Indonesia.value_counts()


# In[7]:


df.Malta.value_counts()


# In[8]:


df.India.value_counts()


# In[9]:


x1=np.array([[271,267,269,280],[29,33,31,20]])
x1


# In[11]:


chi2_contingency(x1)


# In[18]:


tabvalue = stats.chi2.ppf(q=0.95,df=3)


# In[19]:


pvalue =0.27
alpha = 0.05


# In[22]:


alpha = 0.05
print('Significnace=%.5f, p=%.5f' % (alpha, pvalue))
if pvalue > alpha:
    print ("h0 is accepted and h1 is rejected ")
else:
    print ("h1 is accepted and h0 is rejected")


# In[ ]:




