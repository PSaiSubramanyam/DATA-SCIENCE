#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns


# In[54]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report 
from sklearn.metrics import roc_curve


# In[55]:


df=pd.read_csv('bank-full.csv',sep =';' )
df.head()


# In[56]:


df.info()


# In[57]:


df.shape


# In[58]:


df.isnull().sum()


# In[59]:


df[['job','marital','education','default','housing','loan','contact','month','poutcome','y']]=df[['job','marital','education','default','housing','loan','contact','month','poutcome','y']].apply(lambda x: pd.factorize(x)[0])
df              

#converting into dummy variables


# In[60]:


from sklearn.preprocessing import LabelEncoder 
LE = LabelEncoder()
X = df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
Y = df.iloc[:,16] 


# In[61]:


#moel fitting 

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X,Y)


# In[62]:


logreg.coef_


# In[63]:


logreg.predict_proba(X)


# In[64]:


# # Prediction
y_predr = logreg.predict(X)


# In[65]:


df["Y_pred"] = Y_pred
df


# In[66]:


Y_prob = pd.DataFrame(logreg.predict_proba(X.iloc[:,:]))
new_df = pd.concat([df,Y_prob],axis=1)
new_df


# In[67]:


confusion_matrix = confusion_matrix(Y,Y_pred)
print (confusion_matrix)


# In[68]:


pd.crosstab(Y_pred,Y)  


# In[69]:


accuracy = sum(Y==Y_pred)/df.shape[0]
accuracy


# In[70]:


print (classification_report (Y, Y_pred))  


# In[71]:


Logit_roc_score=roc_auc_score(Y,logreg.predict(X))
Logit_roc_score  


# In[72]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.30,random_state=42)


# In[73]:


X_train.shape
Y_train.shape 


# In[74]:


X_test.shape
Y_test.shape


# In[75]:


logreg.fit(X_train,Y_train)


# In[76]:


Y_pred_train = logreg.predict(X_train)
Y_pred_test = logreg.predict(X_test)


# In[77]:


from sklearn.metrics import accuracy_score
print ('training accuracy score:', accuracy_score(Y_train, Y_pred_train).round(2))
print ('testing  accuracy score:', accuracy_score(Y_test, Y_pred_test).round(2))


# In[78]:


##ROC_Curve

fpr, tpr, thresholds = roc_curve(Y,logreg.predict_proba(X)[:,1]) 
plt.plot(fpr, tpr, label='Logistic Regression (area=%0.2f)'% Logit_roc_score)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])                 
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')    
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()            


# In[79]:


y_prob1 = pd.DataFrame(logreg.predict_proba(X)[:,1]) 
y_prob1


# In[ ]:




