#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[27]:


df = pd.read_csv('zoo.csv')


# In[28]:


df


# In[29]:


y=df['type'].values


# In[30]:


X=df.drop(['type','animal name'],axis=1).values


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('type', axis=1), df['type'], test_size=0.25)


# In[32]:


X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)


# In[33]:


knn.score(X_train,y_train)


# In[34]:


knn.score(X_test,y_test)


# In[ ]:




