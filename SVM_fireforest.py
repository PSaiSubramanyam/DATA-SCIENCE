

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

"""# **Problem Statement -> Classify the size of forest using SVM**"""

data = pd.read_csv('forestfires.csv')

"""# **EDA**"""

data.head(5)

data.info()

data.describe()

data.isnull().any()

corr = data[data.columns[0:11]].corr()

sns.heatmap(corr,square=True,annot=True)
plt.figure(figsize=(15,15))

sns.boxplot(data,x='temp',color='g', palette=None, saturation=0.75, width=10 ,dodge=True, fliersize=5)
plt.show()

ax = sns.boxplot(data['area'])

sns.regplot(data,x='temp',y='wind',color='r')

plt.rcParams["figure.figsize"] = 9,5

import warnings
warnings.filterwarnings('ignore')

plt.figure(figsize=(16,5))
print("Skew: {}".format(data['area'].skew()))
print("Kurtosis: {}".format(data['area'].kurtosis()))
ax = sns.kdeplot(data['area'],shade=True,color='g')
plt.xticks([i for i in range(0,1200,50)])
plt.show()

"""From the above graph we can say that data is left skewed and has high kurtosis value."""

dfa = data[data.columns[0:10]]
month_colum = dfa.select_dtypes(include='object').columns.tolist()

plt.figure(figsize=(16,10))
for i,col in enumerate(month_colum,1):
    plt.subplot(2,2,i)
    sns.countplot(data=dfa,y=col)
    plt.subplot(2,2,i+2)
    data[col].value_counts(normalize=True).plot.bar()
    plt.ylabel(col)
    plt.xlabel('% distribution per category')
plt.tight_layout()
plt.show()

"""Majority of the fire was in the month of august and september , most recorded on sundays ans fridays."""

num_columns = dfa.select_dtypes(exclude='object').columns.tolist()

plt.figure(figsize=(18,40))
for i,col in enumerate(num_columns,1):
    plt.subplot(8,4,i)
    sns.kdeplot(data[col],color='g',shade=True)
    plt.subplot(8,4,i+10)
    data[col].plot.box()
plt.tight_layout() 
plt.show()
num_data = data[num_columns]
pd.DataFrame(data=[num_data.skew(),num_data.kurtosis()],index=['skewness','kurtosis'])

X = data.iloc[:,2:30]
y = data.iloc[:,30]

mapping = {'small': 1, 'large': 2}

y = y.replace(mapping)

"""# **SVM**"""

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3,random_state=42)

X_train.shape, y_train.shape, X_test.shape, y_test.shape

model_linear = SVC(kernel = "linear")
model_linear.fit(X_train,y_train)
pred_test_linear = model_linear.predict(X_test)
print("Accuracy:",accuracy_score(y_test, pred_test_linear))
accuracy_linear=accuracy_score(y_test, pred_test_linear)

model_poly = SVC(kernel = "poly")
model_poly.fit(X_train,y_train)
pred_test_poly = model_poly.predict(X_test)
print("Accuracy:",accuracy_score(y_test, pred_test_poly))
accuracy_poly = accuracy_score(y_test, pred_test_poly)

model_rbf = SVC(kernel = "rbf")
model_rbf.fit(X_train,y_train)
pred_test_rbf = model_rbf.predict(X_test)
print("Accuracy:",accuracy_score(y_test, pred_test_rbf))
accuracy_rbf = accuracy_score(y_test, pred_test_rbf)

model_sigmoid = SVC(kernel = "sigmoid")
model_sigmoid.fit(X_train,y_train)
pred_test_sigmoid = model_sigmoid.predict(X_test)
print("Accuracy:",accuracy_score(y_test, pred_test_sigmoid))
accuracy_sigmoid = accuracy_score(y_test, pred_test_sigmoid)

plt.figure(figsize=(15,5))
fn = [accuracy_linear,accuracy_poly,accuracy_rbf,accuracy_sigmoid]
d = ['linear','Poly','RBF','Sigmoid']
plt.bar(x=d,height=fn)

"""From the above graph we can say that Linear SVM has highest accuracy in classifying the forest size"""