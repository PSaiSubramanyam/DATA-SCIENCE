#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sn


# In[6]:


from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import warnings 
warnings.filterwarnings('ignore')


# In[7]:


df=pd.read_csv('crime_data.csv')
df


# In[8]:


df.head()


# In[9]:


df.info()


# In[10]:


crime=df.drop("Unnamed: 0",axis=1)
crime


# In[11]:


# Normalization
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[12]:


df_norm = norm_func(crime.iloc[:,:])
df_norm 


# In[13]:


# #KMEANS Clustering
fig = plt.figure(figsize=(10, 8))
WCSS = []
for i in range(1, 11):
    clf = KMeans(n_clusters=i)
    clf.fit(df_norm)
    WCSS.append(clf.inertia_) # inertia is another name for WCSS
plt.plot(range(1, 11), WCSS)
plt.title('The Elbow Method')
plt.ylabel('WCSS')
plt.xlabel('Number of Clusters')
plt.show()  


# In[14]:


clf = KMeans(n_clusters=5)
y_kmeans = clf.fit_predict(df_norm) 


# In[15]:


clf.labels_ 


# In[16]:


y_kmeans 


# In[17]:


clf.cluster_centers_


# In[18]:


clf.inertia_


# In[19]:


md=pd.Series(y_kmeans)  #converting numpy array into pandas series object 
crime['clust']=md       #creating a  new column and assigning it to new column 
crime


# In[20]:


crime.groupby(crime.clust).mean() 


# In[21]:


WCSS


# In[36]:


## DBSCAN Clustering


# In[37]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


# In[38]:


crime


# In[39]:


array=crime.values
array


# In[40]:


stscaler = StandardScaler().fit(array)
X = stscaler.transform(array) 
X  


# In[41]:


dbscan = DBSCAN(eps=1.25, min_samples=5)
dbscan.fit(X)


# In[42]:


#Noisy samples are given the label -1.
dbscan.labels_


# In[43]:


c=pd.DataFrame(dbscan.labels_,columns=['cluster'])  


# In[44]:


c
pd.set_option("display.max_rows", None)  


# In[45]:


c


# In[47]:


df = pd.concat([df,c],axis=1)  
df     


# In[48]:


d1=dbscan.labels_
d1


# In[49]:


import sklearn
sklearn.metrics.silhouette_score(X, d1) 


# In[50]:


from sklearn.cluster import KMeans
clf = KMeans(n_clusters=5)
y_kmeans = clf.fit_predict(X)


# In[51]:


y_kmeans


# In[52]:


cl1=pd.DataFrame(y_kmeans,columns=['Kcluster']) 
cl1


# In[53]:


df1 = pd.concat([df,cl1],axis=1) 
df1 


# In[55]:


## Silhoutte_score  


# In[56]:


sklearn.metrics.silhouette_score(X, y_kmeans)


# In[57]:


## DBSCAN Visualization


# In[58]:


df.plot(x="Unnamed: 0",y ="cluster",c=dbscan.labels_ ,kind="scatter",s=50 ,cmap=plt.cm.copper_r) 
plt.title('Clusters using DBScan')  


# In[69]:


df1.plot(x="Unnamed: 0",y ="Kcluster",c=y_kmeans ,kind="scatter",s=50 ,cmap=plt.cm.copper_r) 
plt.title('Clusters using KMeans') 


# In[70]:


# #HIERARCHAICAL Clustering
df


# In[71]:


crime


# In[72]:


# ####Standard Scaler


# In[73]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
crime_subset = pd.DataFrame(scaler.fit_transform(crime.iloc[:,1:7]))
crime_subset


# In[75]:


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
    #leaf_font_size=15.,  # font size for the x axis labels)
plt.show()    


# In[77]:


p = np.array(df_norm) # converting into numpy array format 
z = linkage(df_norm, method="average",metric="euclidean")
plt.figure(figsize=(15, 5))
plt.title('DBSCAN Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(z)
    #leaf_rotation=6.,  # rotates the x axis labels
    #leaf_font_size=15.,  # font size for the x axis labels

plt.show()    


# In[80]:


p = np.array(df_norm) # converting into numpy array format 
z = linkage(df_norm, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5))
plt.title('Kmeans Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(z)
    
    #leaf_rotation=6.,  # rotates the x axis labels
    #leaf_font_size=15.,  # font size for the x axis labels

plt.show()    


# In[82]:


from sklearn.cluster import AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=5, linkage='complete',affinity = "euclidean").fit(df_norm) 

cluster_labels=pd.Series(h_complete.labels_)
cluster_labels
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime   


# In[83]:


crime.iloc[:,1:].groupby(crime.clust).mean()


# In[93]:


df0 = crime[(crime.clust==0)]
df0  


# In[92]:


df1 = crime[(crime.clust==1)]
df1


# In[91]:


df2 = crime[(crime.clust==2)]
df2


# In[90]:


df3 = crime[(crime.clust==3)]
df3  


# In[89]:


df4 = crime[(crime.clust==4)]
df4  


# In[ ]:


###Inference
# In Hierarchical cluster, Complete method is suitable for clustering the crime data. 

