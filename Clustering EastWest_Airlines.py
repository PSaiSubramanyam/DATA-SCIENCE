#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


# In[2]:


df = pd.read_excel('EastWestAirlines.xlsx',sheet_name='data')
df


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.dtypes


# In[7]:


df.shape


# In[8]:


air=df.drop(['ID#','Award?'], axis=1)
air


# In[9]:


# ####Normalization 


# In[10]:


def norm_func(i):
  x = (i-i.min())/(i.max()-i.min())
  return (x)


# In[11]:


df_norm = norm_func(air.iloc[:,:])
df_norm 


# In[12]:


# #KMEANS Clustering


# In[13]:


from sklearn.cluster import KMeans


# In[14]:


fig = plt.figure(figsize=(10, 7))
WCSS = []
for i in range(1, 15):
    clf = KMeans(n_clusters=i)
    clf.fit(df_norm)
    WCSS.append(clf.inertia_) # inertia is another name for WCSS
plt.plot(range(1, 15), WCSS)
plt.title('The Elbow Method')
plt.ylabel('WCSS')
plt.xlabel('Number of Clusters')
plt.show()  


# In[15]:


WCSS


# In[16]:


clf = KMeans(n_clusters=4)
y_kmeans = clf.fit_predict(df_norm)  


# In[17]:


y_kmeans


# In[18]:


clf.cluster_centers_ 


# In[19]:


clf.inertia_


# In[20]:


md=pd.Series(y_kmeans)  # converting numpy array into pandas series object 
air['clust']=md # creating a  new column and assigning it to new column 
air


# In[21]:


air.groupby(air.clust).mean() 


# In[22]:


df.plot(x="Balance",y ="Qual_miles",c=y_kmeans ,kind="scatter",s=50 ,cmap=plt.cm.copper_r) 
plt.title('Clusters using KMeans') 


# In[23]:


# #DBSCAN Clustering


# In[24]:


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# In[25]:


air


# In[26]:


array=air.values
array


# In[27]:


stscaler = StandardScaler().fit(array)


# In[28]:


x = stscaler.transform(array)
x


# In[29]:


dbscan = DBSCAN(eps=0.70, min_samples=10)
dbscan.fit(x)


# In[30]:


dbscan.labels_ 


# In[31]:


c=pd.DataFrame(dbscan.labels_,columns=['cluster'])  


# In[32]:


c


# In[33]:


df = pd.concat([df,c],axis=1)  
df   


# In[34]:


d1=dbscan.labels_
d1


# In[35]:


import sklearn
sklearn.metrics.silhouette_score(x, d1)


# In[36]:


from sklearn.cluster import KMeans
clf = KMeans(n_clusters=5)
y_kmeans = clf.fit_predict(x)


# In[37]:


y_kmeans


# In[38]:


cl1=pd.DataFrame(y_kmeans,columns=['Kcluster']) 
cl1


# In[39]:


df1 = pd.concat([df,cl1],axis=1) 
df1 


# In[40]:


# silhouette_score
sklearn.metrics.silhouette_score(x, y_kmeans)


# In[41]:


# DBSCAN Visualization


# In[42]:


df.plot(x="ID#",y ="cluster",c=dbscan.labels_ ,kind="scatter",s=50 ,cmap=plt.cm.copper_r) 
plt.title('Clusters using DBScan')
plt.xlabel("ID#")
plt.ylabel("cluster")


# In[43]:


df1.plot(x="ID#",y ="Kcluster",c=y_kmeans ,kind="scatter",s=50 ,cmap=plt.cm.copper_r) 
plt.title('Clusters using KMeans') 


# In[44]:


# # HIERARCHAICAL Clustering


# In[45]:


df


# In[46]:


air


# In[47]:


air2=df.drop(['ID#','Award?'],axis=1)


# In[48]:


air2


# In[49]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
air2_subset = pd.DataFrame(scaler.fit_transform(air2.iloc[:,1:7]))
air2_subset


# In[50]:


# DENDROGROM


# In[51]:


from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch # for creating dendrogram 
p = np.array(df_norm) # converting into numpy array format 
z = linkage(df_norm, method="single",metric="euclidean")
plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(z)
    
    #leaf_rotation=6.,  # rotates the x axis labels
    #leaf_font_size=15.,  # font size for the x axis labels

plt.show()  


# In[52]:


p = np.array(df_norm) 
z = linkage(df_norm, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(z)
plt.show() 


# In[53]:


p = np.array(df_norm) 
z = linkage(df_norm, method="average",metric="euclidean")
plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(z)
plt.show()    


# In[54]:


from sklearn.cluster import AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=5, linkage='complete',affinity = "euclidean").fit(df_norm) 

cluster_labels=pd.Series(h_complete.labels_)
cluster_labels
air['clust']=cluster_labels   
air


# In[55]:


air.iloc[:,1:].groupby(air.clust).mean()


# In[56]:


data = air[(air.clust==0)]
data  


# In[57]:


data = air[(air.clust==1)]
data  


# In[58]:


data = air[(air.clust==2)]
data  


# In[59]:


data = air[(air.clust==3)]
data  


# In[60]:


data = air[(air.clust==4)]
data  


# In[61]:


# #Inference
# In Hierarichical clustering, complete method is suitable to form cluster for EastWestairlines.

