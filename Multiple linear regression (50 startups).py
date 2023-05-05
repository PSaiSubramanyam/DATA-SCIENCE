#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas  as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('50_Startups.csv')
df


# In[3]:


d1 ={"R&D Spend": "RDSpend","Marketing Spend":"MarketingSpend"}

df.rename(columns=d1,inplace =True)


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.corr()


# In[8]:


Y=df["Profit"]
x1=df["RDSpend"]
x2=df["Administration"]
x3=df["MarketingSpend"]


# In[9]:


# scatter plot for R nd D spend   
plt.scatter(Y,x1)
plt.show()


# In[10]:


# scatter plot for Administration    
plt.scatter(Y,x2)
plt.show()


# In[11]:


# scatter plot  for Marketing Spend    
plt.scatter(Y,x3)
plt.show()


# In[12]:


sns.pairplot(df)


# In[13]:


sns.distplot(df['Profit'])


# In[14]:


## to find the MSE,RMSE AND R^2 value of R&d spend
X1 =df[['RDSpend']]
from  sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X1,Y)


# In[15]:


LR.intercept_ 


# In[16]:


LR.coef_


# In[17]:


Y_pred = LR.predict(X1)


# In[18]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
mse


# In[19]:


rmse = np.sqrt(mse)
rmse


# In[20]:


from sklearn.metrics import r2_score
r2= r2_score(Y,Y_pred)
print ("R square :",r2.round(2))


# In[21]:


import statsmodels.formula.api as smf
model =smf.ols('Profit ~ RDSpend',data=df).fit()
model.summary()


# In[22]:


## to find the MSE,RMSE AND R^2 value of R&d spend,Administration 
X2 =df[['RDSpend','Administration']]
from  sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X2,Y)


# In[23]:


LR.intercept_


# In[24]:


LR.coef_


# In[25]:


Y_pred = LR.predict(X2)


# In[26]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
mse


# In[27]:


rmse = np.sqrt(mse)
rmse


# In[28]:


from sklearn.metrics import r2_score
r2= r2_score(Y,Y_pred)
print ("R square :",r2.round(2))


# In[29]:


import statsmodels.formula.api as smf
model =smf.ols('Profit~RDSpend + Administration',data=df).fit()
model.summary()


# In[30]:


### to find the MSE,RMSE AND R^2 value of R&d spend,Administration,Marketing Spend 

X3 =df[['RDSpend','Administration','MarketingSpend']]
import numpy as np
from  sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X3,Y)


# In[31]:


LR.coef_


# In[32]:


LR.intercept_


# In[33]:


Y_pred = LR.predict(X3)


# In[34]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
mse


# In[35]:


rmse = np.sqrt(mse)
rmse


# In[36]:


from sklearn.metrics import r2_score
r2= r2_score(Y,Y_pred)
print ("R square :",r2.round(2))


# In[37]:


import statsmodels.formula.api as smf
model =smf.ols('Profit~RDSpend + Administration + MarketingSpend',data=df).fit()
model.summary() 


# In[38]:


import statsmodels.formula.api as smf
model =smf.ols('Profit~RDSpend + Administration + MarketingSpend',data=df).fit()
model.summary()


# In[ ]:




