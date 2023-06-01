#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
import seaborn as sn


# In[6]:


df=pd.read_csv("book.csv")
df


# # ### Apriori Algorithm

# In[ ]:


# If support value is 0.1 and threshold is 0.7 
frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
frequent_itemsets


# In[ ]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)
rules
rules.sort_values('lift',ascending = False).head(10)
# (ItalCook)	(CookBks) are at the high confidence of 100%


# In[ ]:


rules.sort_values('lift',ascending = False)[0:20]


# In[ ]:


rules[rules.lift>1]


# In[ ]:


sn.scatterplot(x='support',y='confidence', data= rules)


# # ###If suppose support value is 0.2 and threshold is 1.0

# In[ ]:


frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
frequent_itemsets


# In[ ]:


rules1 = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules1
rules1.sort_values('lift',ascending = False).head(10)
#(ChildBks)	(CookBks) are at the high confidence of 60.5%


# In[ ]:


rules1.sort_values('lift',ascending = False)[0:20]


# In[ ]:


rules1[rules1.lift>1]


# In[ ]:


sn.scatterplot(x='confidence',y='support',data=rules1)


# ## Inference

# In[ ]:


# ###(ItalCook)	(CookBks) are the high confidence  of 100% at support value is 0.1 and threshold value is 0.7.
# ###(ChildBks)	(CookBks) are at the high confidence of 60.5% at support value is 0.2 and threshold value is 1.

