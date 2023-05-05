#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 


# In[5]:


df = pd.read_csv("delivery_time.csv")
df


# In[6]:


df.head()


# In[8]:


df.shape


# In[9]:


df.describe()


# In[11]:


df.dtypes


# In[22]:


df['Delivery Time'].value_counts()


# In[23]:


df['Sorting Time'].value_counts()


# In[12]:


Y=df["Delivery Time"]
X=df[["Sorting Time"]]


# In[13]:


import matplotlib.pyplot as plt
plt.scatter(Y,X)
plt.show()


# In[1]:


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


# In[25]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
mse


# In[32]:


import numpy as np
print("mean square error:",np.sqrt(mse).round(2)) 


# In[31]:


import statsmodels.formula.api as smf
model =smf.ols('Y ~ X',data=df).fit()
model.summary()


# In[ ]:




