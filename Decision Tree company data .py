#!/usr/bin/env python
# coding: utf-8

# In[67]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets


# In[68]:


from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing


# In[69]:


df = pd.read_csv('Company_Data.csv')
df


# In[70]:


df.info


# In[71]:


df.head()


# In[72]:


df.shape


# In[73]:


sns.pairplot(df)


# In[74]:


plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),annot=True)


# In[75]:


label_encoder = preprocessing.LabelEncoder()
df['ShelveLoc']= label_encoder.fit_transform(df['ShelveLoc'])
df['Urban']= label_encoder.fit_transform(df['Urban'])
df['US']= label_encoder.fit_transform(df['US'])


# In[76]:


df


# In[77]:


x=df.iloc[:,0:6]
y=df['ShelveLoc']


# In[78]:


x


# In[79]:


y


# In[80]:


df['ShelveLoc'].unique() 


# In[81]:


df.ShelveLoc.value_counts()


# In[82]:


colnames = list(df.columns)
colnames


# In[83]:


x_train,x_test,y_train,y_test = train_test_split(x,y)


# In[84]:


DT = DecisionTreeClassifier(criterion='gini',max_depth=8)


# In[85]:


DT.fit(x_train,y_train)


# In[86]:


DT.tree_.node_count  # counting the number of nodes 
DT.tree_.max_depth   # number of levels  


# In[87]:


Y_pred_train = DT.predict(x_train)
Y_pred_test = DT.predict(x_test)


# In[88]:


ac1 = accuracy_score(y_train, Y_pred_train)
print ('training accuracy score:',ac1.round(3))


# In[89]:


ac2= accuracy_score(y_test, Y_pred_test)
print ('testing  accuracy score:', ac2.round(3))


# In[90]:


tree.plot_tree(DT);             #PLot the decision tree


# In[91]:


fn=['Sales','CompPrice', 'Income','Advertising','Population','Price']
cn=['Bad', 'Good', 'Medium']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(DT,
               feature_names = fn, 
               class_names=cn,
               filled = True);


# In[92]:


DT.feature_importances_


# In[93]:


feature_imp = pd.Series(DT.feature_importances_,index=fn).sort_values(ascending=False) 
feature_imp


# In[94]:


sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()


# In[95]:


#Predicting on test data

preds = DT.predict(x_test)                  # predicting on test data set 
pd.Series(preds).value_counts()                # getting the count of each category 


# In[96]:


preds


# In[97]:


pd.crosstab(y_test,preds)  # getting the 2 way table to understand the correct and wrong predictions


# In[98]:


np.mean(preds==y_test)          #accuracy


# # #Building Decision Tree Classifier (CART) using Gini Criteria
# # 

# In[99]:


DT_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)


# In[100]:


DT_gini.fit(x_train, y_train) 


# In[101]:


pred=DT.predict(x_test)
np.mean(preds==y_test)                       #Prediction and computing the accuracy 


# In[102]:


DT.feature_importances_ 


# # #Bagging regressor
# 

# In[103]:


from sklearn.ensemble import BaggingClassifier
bag= BaggingClassifier(base_estimator=DT,n_estimators=50,max_samples=0.5, max_features=0.6)


# In[104]:


bag.fit(x_train,y_train)
Y_pred_train = bag.predict(x_train)
Y_pred_test = bag.predict(x_test) 


# In[105]:


import numpy as np
from sklearn.metrics import mean_squared_error


# In[106]:


mse1 = np.sqrt(mean_squared_error(y_train, Y_pred_train))
print("training error:",mse1.round(3))


# In[107]:


mse2 = np.sqrt(mean_squared_error(y_test, Y_pred_test))
print("test error:",mse2.round(3))


# In[108]:


print("bagging-difference of tain and test: ", (mse2-mse1).round(2))


# # #Decision Tree Regression

# In[109]:


from sklearn.tree import DecisionTreeRegressor 


# In[110]:


array = df.values
X = array[:,0:6]
y = array[:,3] 


# In[111]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1) 


# In[112]:


DT = DecisionTreeRegressor()
DT.fit(x_train, y_train) 


# In[113]:


DT.score(x_test,y_test)           #accuracy


# In[ ]:




