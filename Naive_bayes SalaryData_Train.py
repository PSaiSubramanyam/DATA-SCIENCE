#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('SalaryData_Train.csv')
df


# In[4]:


df.info()


# In[5]:


df.describe()


# In[9]:


df.dtypes


# In[10]:


correlation = df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')       
plt.title('Correlation between different fearures')


# In[11]:


df.describe().round(2).style.background_gradient(cmap = 'Blues')


# In[12]:


sns.heatmap(df.isnull(),cmap='Blues')


# In[13]:


sns.pairplot(df)


# In[14]:


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()


# In[15]:


for i in range (0,14):
    df.iloc[:,i] = LE.fit_transform(df.iloc[:,i])


# In[16]:


df.head()


# In[18]:


df[['workclass','education','maritalstatus','occupation','relationship','race','sex','native','Salary']] = df[['workclass','education','maritalstatus','occupation','relationship','race','sex','native','Salary']].apply(lambda x: pd.factorize(x)[0])
df


# In[19]:


X=df.iloc[:,0:13]
Y=df.iloc[:,13]


# In[20]:


X


# In[21]:


Y


# In[22]:


from sklearn.naive_bayes import MultinomialNB as MB


# In[23]:


classifier_mb = MB()
classifier_mb.fit(X,Y)


# In[24]:


train_pred_m = classifier_mb.predict(X)
accuracy_train_m = np.mean(train_pred_m==Y)


# In[26]:


print('Training accuracy is:',accuracy_train_m)


# In[27]:


#data partition 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.30,random_state=42)


# In[34]:


from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
MNB.fit(X_train,Y_train)


# In[35]:


Y_pred_train = MNB.predict(X_train)
Y_pred_test = MNB.predict(X_test) 


# In[36]:


from sklearn import metrics


# In[37]:


print ('Training accuracy score:', metrics.accuracy_score(Y_train, Y_pred_train).round(2))


# In[38]:


print ('Testing  accuracy score:', metrics.accuracy_score(Y_test, Y_pred_test).round(2))


# In[39]:


# ###Gaussian Naive Bayes


# In[40]:


from sklearn.naive_bayes import GaussianNB as GB


# In[41]:


classifier_gb = GB()
classifier_gb.fit(X,Y) 


# In[42]:


train_pred_g = classifier_gb.predict(X)
accuracy_train_g = np.mean(train_pred_g==Y)


# In[43]:


print('Training accuracy is:',accuracy_train_g)


# In[44]:


# with KNN 
from sklearn.neighbors import  KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)


# In[45]:


knn.fit(X_train,Y_train)


# In[46]:


Y_pred_train = knn.predict(X_train)
Y_pred_test  = knn.predict(X_test)


# In[48]:


for i in range (0,13):
    df.iloc[:,i] = LE.fit_transform(df.iloc[:,i])


# In[49]:


from sklearn import metrics


# In[50]:


print ('training accuracy score:', metrics.accuracy_score(Y_train, Y_pred_train).round(2))


# In[51]:


print ('testing  accuracy score:', metrics.accuracy_score(Y_test, Y_pred_test).round(2))


# In[ ]:




