#!/usr/bin/env python
# coding: utf-8

# In[109]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[110]:


df=pd.read_csv('SalaryData_Test.csv')
df


# In[111]:


df.info


# In[112]:


df.describe()


# In[113]:


df.describe().round(2).style.background_gradient(cmap = 'Reds')


# In[114]:


df.dtypes


# In[115]:


correlation = df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')       
plt.title('Correlation between different fearures')


# In[116]:


sns.heatmap(df.isnull(),cmap='Reds')


# In[117]:


sns.pairplot(df)


# In[118]:


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()


# In[119]:


for i in range (0,13):
    df.iloc[:,i] = LE.fit_transform(df.iloc[:,i])


# In[120]:


df.head()


# In[121]:


x=df.iloc[:,0:13]
y=df.iloc[:,13:]


# In[122]:


x


# In[123]:


y


# ##Naive Bayes

# In[124]:


#data partition 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.30,random_state=42)


# In[125]:


from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
MNB.fit(x_train,y_train)
Y_pred_train = MNB.predict(x_train)
Y_pred_test = MNB.predict(x_test) 


# In[126]:


from sklearn import metrics


# In[127]:


print ('training accuracy score:', metrics.accuracy_score(y_train, Y_pred_train).round(2))


# In[128]:


print ('testing  accuracy score:', metrics.accuracy_score(y_test, Y_pred_test).round(2))


# In[129]:


##  by using KNN 


# In[130]:


from sklearn.neighbors import  KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)


# In[131]:


knn.fit(x_train,y_train)


# In[132]:


Y_pred_train = knn.predict(x_train)
Y_pred_test  = knn.predict(x_test)


# In[133]:


for i in range (0,13):
    df.iloc[:,i] = LE.fit_transform(df.iloc[:,i])


# In[134]:


from sklearn import metrics


# In[135]:


print ('training accuracy score:', metrics.accuracy_score(y_train, Y_pred_train).round(2))


# In[136]:


print ('testing  accuracy score:', metrics.accuracy_score(y_test, Y_pred_test).round(2))

