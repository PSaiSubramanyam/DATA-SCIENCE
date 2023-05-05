#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.stats as stats 
import pandas as pd 
import numpy as np 


# In[2]:


df=pd.read_csv("Cutlets.csv")


# In[3]:


df.head(10)


# In[4]:


df.describe()


# In[5]:


df.isnull().sum()


# In[6]:


df[df.duplicated()].shape


# In[7]:


df.info()


# In[8]:


#now to take the samples that we using functions  

import statsmodels.api as sm


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


# THE SAMPLE OF BOX PLOT 
plt.subplots(figsize = (8,5))
plt.subplot(121)
plt.boxplot(df['Unit A'])
plt.title('Unit A')
plt.subplot(122)
plt.boxplot(df['Unit B'])
plt.title('Unit B')
plt.show()


# In[11]:


# THE SAMPLE OF HISTORAGM 
plt.subplots(figsize = (8,5))
plt.subplot(121)
plt.hist(df['Unit A'],)
plt.title('Unit A')
plt.subplot(122)
plt.hist(df['Unit B'],)
plt.title('Unit B')
plt.show()


# In[12]:


import warnings
warnings.filterwarnings("ignore")
from PIL import ImageGrab


# In[13]:


#THE SAMPLE OF DISTRIBUTIVE PLOT 
plt.figure(figsize = (8,5))
labels = ['Unit A', 'Unit B']
sns.distplot(df['Unit A'], kde= True)   #kde= kernel density 
sns.distplot(df['Unit B'],hist = True)
plt.legend(labels)


# In[14]:


# THE SAMPLE OF QUANTILE - QUANTILE PLOT 
sm.qqplot(df["Unit A"], line = 'q')
plt.title('Unit A')
sm.qqplot(df["Unit B"], line = 'q')
plt.title('Unit B')
plt.show()


# In[15]:


statistic , p_value = stats.ttest_ind(df['Unit A'],df['Unit B'], alternative = 'two-sided')
print('p_value=',p_value)


# In[17]:


alpha = 0.05
print('Significnace=%.5f, p=%.5f' % (alpha, p_value))
if p_value <= alpha:
     print ("ho is rejected and h1 is accepted")
else:
    print ("h1 is accepted and ho is rejected")


# In[ ]:


#Assumptions:
#The samples are randomly selected.
#The data  not  follows a  normal distribution.
#The variances of the data are not equal.


# In[ ]:




