#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[43]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[44]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn import preprocessing


# In[45]:


df=pd.read_csv('Fraud_check.csv')
df


# In[46]:


df.info()


# In[47]:


sns.pairplot(df)


# In[48]:


sns.heatmap(df.isnull(),cmap='Reds') 


# In[49]:


plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),annot=True)


# In[50]:


label_encoder = preprocessing.LabelEncoder()
df['Undergrad']= label_encoder.fit_transform(df['Undergrad'])
df['Urban']= label_encoder.fit_transform(df['Urban'])
df['Marital.Status']= label_encoder.fit_transform(df['Marital.Status'])


# In[51]:


df


# In[52]:


df['Status'] = df['Taxable.Income'].apply(lambda Income: 'Risky' if Income <= 30000 else 'Good')


# In[53]:


df['Status']= label_encoder.fit_transform(df['Status'])


# In[54]:


df


# In[55]:


df.Status.unique()


# In[56]:


x=df.iloc[:,0:4]
y=df['Status']


# # #Bagged Decision Trees for Classification

# In[57]:


num_trees = 100
seed=8
kfold = KFold(n_splits=100, shuffle = True, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, x,y, cv=kfold)
print(results.mean())


# In[58]:


# #Stacking Ensemble for Classification

kfold = KFold(n_splits=10,shuffle=True, random_state=8)
estimators = []
model1 = LogisticRegression(max_iter=100)                          # create the sub models
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
ensemble = VotingClassifier(estimators)                            # create the ensemble model
results = cross_val_score(ensemble, x, y, cv=kfold)
print(results.mean())


# # #Random Forest Classification

# In[59]:


num_trees = 100
max_features = 3
kfold = KFold(n_splits=10, shuffle= True ,random_state=8)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, x, y, cv=kfold)
print(results.mean())


# In[60]:


# #Boost Classification
num_trees = 100
seed=8
kfold = KFold(n_splits=100, shuffle = True, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, x,y, cv=kfold)
print(results.mean())


# In[ ]:




