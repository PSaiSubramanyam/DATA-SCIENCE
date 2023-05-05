#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.stats as stats
import statsmodels.api as sm
import numpy as np
import pandas as pd


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df1= pd.read_csv('LabTAT.csv')


# In[4]:


df1


# In[5]:


df1.describe()


# In[6]:


df1.isnull().sum()


# In[7]:


df1 [df1.duplicated()].shape 


# In[8]:


df1.info()


# In[9]:



# Box plot for the laboratory 1, 2,3,4. 

plt.subplots(figsize = (16,9))
plt.subplot(221)
plt.boxplot(df1['Laboratory 1'])
plt.title('Laboratory 1')
plt.subplot(222)
plt.boxplot(df1['Laboratory 2'])
plt.title('Laboratory 2')
plt.subplot(223)
plt.boxplot(df1['Laboratory 3'])
plt.title('Laboratory 3')
plt.subplot(224)
plt.boxplot(df1['Laboratory 4'])
plt.title('Laboratory 4')
plt.show()


# In[10]:


plt.subplots(figsize = (9,6))
plt.subplot(221)
plt.hist(df1['Laboratory 1'],color='red')
plt.title('Laboratory 1')
plt.subplot(222)
plt.hist(df1['Laboratory 2'],color='black')
plt.title('Laboratory 2')
plt.subplot(223)
plt.hist(df1['Laboratory 3'],color= 'yellow')
plt.title('Laboratory 3')
plt.subplot(224)
plt.hist(df1['Laboratory 4'],color='green')
plt.title('Laboratory 4')
plt.show()


# In[11]:


import warnings
warnings.filterwarnings("ignore")
from PIL import ImageGrab


# In[12]:


plt.figure(figsize = (8,6))
labels = ['Lab 1', 'Lab 2','Lab 3', 'Lab 4']
sns.distplot(df1['Laboratory 1'])
sns.distplot(df1['Laboratory 2'])
sns.distplot(df1['Laboratory 3'])
sns.distplot(df1['Laboratory 4'])
plt.legend(labels)


# In[13]:


plt.figure(figsize = (4,4))
sm.qqplot(df1['Laboratory 1'])
plt.title('Laboratory 1')
sm.qqplot(df1['Laboratory 2'])
plt.title('Laboratory 2')
sm.qqplot(df1['Laboratory 3'])
plt.title('Laboratory 3')
sm.qqplot(df1['Laboratory 4'])
plt.title('Laboratory 4')
plt.show()


# In[14]:


test_statistic , p_value = stats.f_oneway(df1.iloc[:,0],df1.iloc[:,1],df1.iloc[:,2],df1.iloc[:,3])
print('p_value =',p_value)


# In[15]:


alpha = 0.05
print('Significnace=%.3f, p=%.3f' % (alpha, p_value))
if p_value <= alpha:
    print ("ho is rejected and h1 is accepted")            # ho= Null hypothesis 
else:
    print ("h1 is accepted and ho is rejected")            # h1 = alteranate hypothesis 


# In[ ]:




