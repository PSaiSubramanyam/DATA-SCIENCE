#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[2]:


df=pd.read_csv("Salary_Data.csv")


# In[3]:


df


# In[4]:


df.info()


# In[6]:


df.shape


# In[7]:


df.describe()


# In[8]:


df.dtypes


# In[9]:


df['YearsExperience'].value_counts


# In[10]:


df['Salary'].value_counts


# In[14]:


Y=df["Salary"]
X=df[["YearsExperience"]]


# In[15]:


import matplotlib.pyplot as plt
plt.scatter(Y,X)
plt.show()


# In[16]:


from  sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[17]:


LR.intercept_


# In[18]:


LR.coef_


# In[19]:


Y_pred = LR.predict(X)


# In[20]:


import matplotlib.pyplot as plt 
plt.scatter(X,Y)
plt.scatter(X,Y_pred,color='red')
plt.plot(X,Y_pred,color='black')
plt.show()


# In[21]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
mse


# In[22]:


import numpy as np
print("mean square error:",np.sqrt(mse).round(2)) 


# In[23]:


import statsmodels.formula.api as smf
model =smf.ols('Y ~ X',data=df).fit()
model.summary()


# In[ ]:




