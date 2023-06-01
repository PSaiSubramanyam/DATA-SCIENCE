#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets  
import numpy as np
from sklearn.model_selection import train_test_split


# In[2]:


from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing


# In[3]:


df = pd.read_csv("Fraud_check.csv")


# In[4]:


df


# In[5]:


df.head()


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df=pd.get_dummies(df,columns=['Undergrad','Marital.Status','Urban'], drop_first=True)


# In[10]:


df["TaxInc"] = pd.cut(df["Taxable.Income"], bins = [10002,30000,99620], labels = ["Risky", "Good"])


# In[12]:


print (df)


# ### Lets assume: taxable_income <= 30000 as “Risky=0” and others are “Good=1”

# In[13]:


df = pd.get_dummies(df,columns = ["TaxInc"],drop_first=True)


# In[14]:


df.tail(10)


# In[15]:


import seaborn as sns
sns.pairplot(data=df, hue = 'TaxInc_Good')


# In[16]:


# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[17]:


# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df.iloc[:,1:])
df_norm.tail(10)


# In[18]:


# Declaring features & target
X = df_norm.drop(['TaxInc_Good'], axis=1)
y = df_norm['TaxInc_Good']


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


# Splitting data into train & test
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)


# In[21]:


df_norm["income"]="<=30000"
df_norm.loc[df["Taxable.Income"]>=30000,"income"]="Good"
df_norm.loc[df["Taxable.Income"]<=30000,"income"]="Risky"


# In[22]:


df.drop(["Taxable.Income"],axis=1,inplace=True)


# In[24]:


df.rename(columns={"Undergrad":"undergrad","Marital.Status":"marital","City.Population":"population","Work.Experience":"experience"})


# In[25]:


from sklearn import preprocessing
le=preprocessing.LabelEncoder()
for column_name in df.columns:
    if df[column_name].dtype == object:
        df[column_name] = le.fit_transform(df[column_name])
    else:
        pass


# In[26]:


features = df.iloc[:,0:5]
labels = df.iloc[:,5]


# In[27]:


colnames = list(df.columns)
predictors = colnames[0:5]
target = colnames[5]


# In[28]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size = 0.2,stratify = labels)


# In[29]:


from sklearn.ensemble import RandomForestClassifier as RF
model = RF(n_jobs = 3,n_estimators = 15, oob_score = True, criterion = "entropy")
model.fit(x_train,y_train)


# In[30]:


model.estimators_
model.classes_
model.n_features_
model.n_classes_


# In[31]:


model.n_outputs_


# In[32]:


model.oob_score_


# In[33]:


prediction = model.predict(x_train)


# In[34]:


from sklearn.metrics import accuracy_score


# In[35]:


accuracy = accuracy_score(y_train,prediction)


# In[36]:


np.mean(prediction == y_train)


# In[37]:


##Confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_train,prediction)


# In[38]:


##Prediction on test data
pred_test = model.predict(x_test)


# In[40]:


acc_test =accuracy_score(y_test,pred_test)


# In[41]:


acc_test


# ###Building Decision Tree Classifier

# In[45]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train)


# In[46]:


from sklearn import tree


# In[47]:


tree.plot_tree(model);


# In[48]:


colnames = list(df.columns)
colnames


# In[49]:


fn=['population','experience','Undergrad_YES','Marital.Status_Married','Marital.Status_Single','Urban_YES']
cn=['1', '0']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);


# In[50]:


preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 


# In[51]:


preds


# In[52]:


pd.crosstab(y_test,preds)


# In[53]:


np.mean(preds==y_test)


# Building Decision Tree Classifier (CART) using Gini Criteria

# In[54]:


from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)


# In[55]:


model_gini.fit(x_train, y_train)


# In[58]:


pred=model.predict(x_test)


# In[57]:


np.mean(preds==y_test)


# Decision Tree Regression Example

# In[59]:


from sklearn.tree import DecisionTreeRegressor


# In[60]:


array = df.values
X = array[:,0:3]
y = array[:,3]


# In[61]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# In[62]:


model = DecisionTreeRegressor()
model.fit(X_train, y_train)


# In[63]:


model.score(X_test,y_test)


# In[ ]:




