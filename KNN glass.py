#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[28]:


df = pd.read_csv('glass.csv')


# In[29]:


df


# In[30]:


df.head()


# In[31]:


df.info()


# In[32]:


X = np.array(df.iloc[:,3:5])
Y = np.array(df['Type'])
print("Shape of X:"+str(X.shape))
print("Shape of y:"+str(y.shape))


# In[ ]:





# In[33]:


X_train,X_test,Y_train,Y_test = train_test_split(X,y)
print("Shape of X_Train:"+str(X_train.shape))
print("Shape of y_Train:"+str(Y_train.shape))
print("Shape of X_Test:"+str(X_test.shape))
print("Shape of y_Test:"+str(Y_test.shape))


# In[34]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)


# In[35]:


pred = knn.predict(X_train)
pred


# In[36]:


accuracy = knn.score(X_train,Y_train)
print("The accuracy is :"+str(accuracy))


# In[37]:


pred1 = knn.predict(X_test)
pred1


# In[38]:


accuracy = knn.score(X_test,Y_test)
print("The accuracy is :"+str(accuracy))


# In[ ]:




