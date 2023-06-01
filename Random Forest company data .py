#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[22]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


# In[23]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")


# In[24]:


data = pd.read_csv('Company_Data.csv')
data


# In[25]:


data.info()


# In[26]:


sns.pairplot(data)


# In[27]:


sns.heatmap(data.isnull(),cmap='Blues')


# In[28]:


plt.figure(figsize=(20,10))
sns.heatmap(data.corr(),annot=True)


# In[29]:


get_ipython().system('pip install category_encoders')


# In[30]:


from category_encoders import OrdinalEncoder
encoder = OrdinalEncoder(cols=["ShelveLoc", "Urban", "US"])
sales = encoder.fit_transform(data)


# In[31]:


sale_val = []
for value in data['Sales']:
      if value <= 7.49:
          sale_val.append("low")
      else:
          sale_val.append("high")
sales["sale_val"]= sale_val


# In[32]:


sales


# In[33]:


x = sales.drop(['sale_val', 'Sales'],axis=1)
x


# In[34]:


y= sales['sale_val']
y


# # ## Bagged Decision Trees for Classification

# In[35]:


kfold = KFold(n_splits=10,shuffle = True, random_state = 8)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=8)
results = cross_val_score(model, x,y, cv=kfold)
print(results.mean()) 


# # #Random Forest Classification

# In[36]:


num_trees = 100
max_features = 3
kfold = KFold(n_splits=10, shuffle= True ,random_state=8)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, x, y, cv=kfold)
print(results.mean()) 


# # #Boost Classification

# In[37]:


num_trees = 100
seed=8
kfold = KFold(n_splits=100, shuffle = True, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, x,y, cv=kfold)
print(results.mean())


# # #Stacking Ensemble for Classification

# In[38]:


kfold = KFold(n_splits=10,shuffle=True, random_state=8)
estimators = []
model1 = LogisticRegression(max_iter=100) # create the sub models
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
ensemble = VotingClassifier(estimators) # create the ensemble model
results = cross_val_score(ensemble, x, y, cv=kfold)
print(results.mean())


# In[ ]:




