#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import matplotlib as plt


# In[22]:


df=pd.read_csv('book.csv',encoding='latin-1')
df


# In[25]:


df =df.drop(['Unnamed: 0'],axis=1)


# In[26]:


df1 =df.rename({'User.ID':'user_id','Book.Title':'book_title','Book.Rating':'book_rating'},axis=1)


# In[27]:


df1


# In[28]:


df1.info()


# In[37]:


len(df1.user_id.unique())


# In[38]:


len(df1.book_title.unique())


# In[45]:


df2 = df1.drop_duplicates(['user_id','book_title'])


# In[46]:


books = df2.pivot(index='user_id',columns='book_title', values='book_rating').reset_index(drop=True)
books


# In[47]:


books.index = df2.user_id.unique()


# In[48]:


books


# In[49]:


books.fillna(0, inplace=True)


# In[50]:


books


# In[51]:


from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation


# In[52]:


df3 = 1 - pairwise_distances( books.values,metric='cosine')


# In[53]:


df3


# In[54]:


#Store the results in a dataframe
books2 = pd.DataFrame(df3)


# In[58]:


books2.index = df1.user_id.unique()
books2.columns = df1.user_id.unique()


# In[59]:


books2.iloc[0:5, 0:5]


# In[61]:


np.fill_diagonal(df3, 0)
books2.iloc[0:5, 0:5]


# In[62]:


books2.idxmax(axis=1)[0:5]


# In[ ]:




