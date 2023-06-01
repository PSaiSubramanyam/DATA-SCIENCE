import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

"""**Problem Statement-->Perform PCA and perform clustering using first 3 principal component scores using hierarchy and K-means clustering**

**EDA**
"""

df = pd.read_csv("/content/wine.csv")
df.head()

df.info()

df.describe()

df.isnull().sum()

df.duplicated().sum()

df1 = df.iloc[:,1:]
df1

df1.corr()

# Converting into numpy array
Data = df1.values
Data

# Normalizing the numerical data 
data_normal = scale(Data)
data_normal

"""# **Principal Component Analysis**"""

pca = PCA(n_components = 13)
pca_values = pca.fit_transform(data_normal)
pca_values

pca.components_

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_ 
var

# Cumulative variance 
var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1

plt.plot(var1,color="red")

"""# Plotting graph between the PCA variables"""

PC = range(1, pca.n_components_+1)
plt.bar(PC, pca.explained_variance_ratio_, color='blue')
plt.xlabel('Principal Components')
plt.ylabel('Variance %')
plt.xticks(PC)
import warnings
warnings.filterwarnings('ignore')

PCA_components = pd.DataFrame(pca_values)

plt.scatter(PCA_components[0], PCA_components[1], alpha=0.5, color='blue')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

finalDf = pd.concat([pd.DataFrame(pca_values[:,0:2],columns=['pc1','pc2']), df[['Type']]], axis = 1)
finalDf

import seaborn as sns
sns.scatterplot(data=finalDf,x='pc1',y='pc2',hue='Type',s = 100)

"""# From the above graph we can clearly say that there are 3 differnet types of clusters

# **Hierarchy clustering**
"""

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

model1 = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')

h_cluster = model1.fit(PCA_components.iloc[:,:2])

labels1 = model1.labels_
labels1

X = PCA_components.iloc[:,:1]
Y = PCA_components.iloc[:,1:2]
X

Y

plt.figure(figsize=(15, 5))  
plt.scatter(X, Y, c=labels1)

"""# The graph also indicates three different types of clusters"""

h_df=pd.DataFrame(pca_values[:,0:2])
h_df

hcf = linkage(h_df,method="complete",metric="euclidean")

plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    hcf,
    leaf_rotation=0.,
    leaf_font_size=8.,
)
plt.show()

wcss = []

from sklearn.cluster import KMeans
fig = plt.figure(figsize=(10, 8))
WCSS = [] #within cluster sum of square values in the empty list
for i in range(1, 11):
    clf = KMeans(n_clusters=i)
    clf.fit(data_normal)
    WCSS.append(clf.inertia_) # inertia is another name for WCSS
plt.plot(range(1, 11), WCSS)
plt.title('The Elbow Method')
plt.ylabel('WCSS')
plt.xlabel('Number of Clusters')
plt.show()

"""# The scree plot levels off at K=3 so lets us consider k_nclusters as 3"""

model2 = KMeans(n_clusters=3)
model2.fit(PCA_components.iloc[:,:2])

labels = model2.predict(PCA_components.iloc[:,:2])

plt.scatter(PCA_components[0], PCA_components[1], c=labels)
plt.show()

k_df=pd.DataFrame(pca_values[:,0:2])
k_df

model3 = KMeans(n_clusters=3)
model3.fit(k_df)

model3.labels_

md=pd.Series(model3.labels_)

df1['clust']=md
df1.head()

df1.groupby(df1.clust).mean()

df.plot(x="Alcohol",y ="Hue",c=clf.labels_,kind="scatter",s=30 ,cmap=plt.cm.coolwarm) 
plt.title('Clusters using KMeans')

"""# **Conclusion**

# From Dimensional reduction analysis we have reduced the varibales from 13-->2 and by using the two important PCA components we had done clustering. It clearly shows that, the data set have three different types of wine clusters in it.
"""