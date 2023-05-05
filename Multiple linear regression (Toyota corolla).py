#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm 
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.formula.api as smf
import numpy as np 


# In[2]:


df=pd.read_csv('ToyotaCorolla.csv', encoding= 'unicode_escape') 
df


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.isna().sum()


# In[6]:


df.corr()


# In[9]:


model = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=df).fit()


# In[10]:


model.params


# In[11]:


print(model.tvalues, '\n', model.pvalues)


# In[12]:


(model.rsquared,model.rsquared_adj)


# In[13]:


model.summary()


# In[14]:


ml_cc=smf.ols('Price~cc',data = df).fit()  
print(ml_cc.tvalues, '\n', ml_cc.pvalues)


# In[15]:


ml_cc.summary()


# In[17]:


ml_d=smf.ols('Price~Doors',data = df).fit()  
print(ml_d.tvalues, '\n', ml_d.pvalues) 


# In[18]:


ml_d.summary()


# In[19]:


qqplot=sm.qqplot(model.resid,line='q')
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[20]:


list(np.where(model.resid>2100)) 


# In[21]:


model_influence = model.get_influence()
(c, _) = model_influence.cooks_distance


# In[25]:


fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(df)), np.round(c, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[26]:


(np.argmax(c),np.max(c))


# In[27]:


influence_plot(model)
plt.show()


# In[29]:


k = df.shape[1]
n = df.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff


# In[30]:


df[df.index.isin([80,221,960])]


# In[31]:


df.head()


# In[33]:


df_new=pd.read_csv('ToyotaCorolla.csv', encoding= 'unicode_escape')


# In[34]:


df1=toyota_new.drop(toyota_new.index[[80,221,960]],axis=0).reset_index()
df1.shape


# In[36]:


df2=df1.drop(['index'],axis=1)
df2.shape


# In[37]:


final_ml_V= smf.ols('Price~Age_08_04+KM+HP+cc+Gears+Doors+Quarterly_Tax+Weight',data =df2).fit()


# In[38]:


model_influence_V = final_ml_V.get_influence()
(c_V, _) = model_influence_V.cooks_distance


# In[39]:



fig= plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(df2)),np.round(c_V,3));
plt.xlabel('Row index')
plt.ylabel('Cooks Distance');


# In[40]:


(np.argmax(c_V),np.max(c_V))


# In[41]:



final_ml_V= smf.ols('Price~Age_08_04+KM+HP+cc+Gears+Doors+Quarterly_Tax+Weight',data =df2).fit()


# In[42]:


(final_ml_V.rsquared,final_ml_V.aic)


# In[44]:


new_data=pd.DataFrame({'Age_08_04':50,"KM":160,"HP":1100,"cc":225,"Gears":7,"Weight":250,"Doors":4,"Quarterly_Tax":350},index=[1])


# In[46]:


final_ml_V.predict(new_data)

