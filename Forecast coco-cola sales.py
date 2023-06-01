#!/usr/bin/env python
# coding: utf-8

# In[80]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models
from datetime import datetime,time


# In[81]:


import warnings
import itertools
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
from pylab import rcParams
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
import statsmodels.formula.api as smf


# ## Problem Statement -> Forecast coco-cola sales

# In[82]:


df = pd.read_excel('CocaCola_Sales_Rawdata.xlsx')


# In[83]:


df


# In[84]:


df.head()


# In[85]:


df.info()


# In[86]:


df.describe()


# In[87]:


df.isnull().sum()


# In[88]:


temp = df.Quarter.str.replace(r'(Q\d)_(\d+)', r'19\2-\1')


# In[89]:


df['Month_Year'] = pd.to_datetime(temp).dt.strftime('%b-%Y')


# In[90]:


df.sample(4)


# In[91]:


data = df.drop(['Quarter'], axis=1)


# In[92]:


data.reset_index(inplace=True)


# In[93]:


data['Month_Year'] = pd.to_datetime(data['Month_Year'])


# In[94]:


data = data.set_index('Month_Year')


# In[95]:


data.head()


# In[96]:


data['Sales'].plot(figsize=(15, 6))
plt.show()


# In[97]:


for i in range(2,10,2):
    data["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)


# In[98]:


plt.figure(figsize=(20,15))
ts_add = seasonal_decompose(data.Sales,model="additive")
fig = ts_add.plot()
plt.show()


# In[99]:


plt.figure(figsize=(20,15))
ts_mul = seasonal_decompose(data.Sales,model="multiplicative")
fig = ts_mul.plot()
plt.show()


# In[100]:


tsa_plots.plot_acf(data.Sales)


# ## Model building ARIMA

# In[101]:


X = data['Sales'].values


# In[102]:


size = int(len(X) * 0.66)
size


# In[103]:


train, test = X[0:size], X[size:len(X)]


# In[104]:


from statsmodels.tsa.arima.model import ARIMA


# In[105]:


model = ARIMA(train, order=(5,1,0))


# In[106]:


model_fit = model.fit()


# In[107]:


model_fit.summary()


# In[108]:


residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())


# ## Rolling ARIMA Model

# In[109]:


history = [x for x in train]


# In[110]:


predictions = list()


# In[111]:


for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))


# In[112]:


pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()


# In[113]:


df1 = pd.get_dummies(df, columns = ['Quarter'])


# In[114]:


df1.columns = ['Sales','Q1','Q1','Q1','Q1','Q1','Q1','Q1','Q1','Q1','Q1','Q1','Q2','Q2','Q2','Q2','Q2','Q2','Q2','Q2','Q2','Q2','Q2','Q3','Q3','Q3','Q3','Q3','Q3','Q3','Q3','Q3','Q3','Q4','Q4','Q4','Q4','Q4','Q4','Q4','Q4','Q4','Q4','Q4']


# In[115]:


df1.head()


# In[116]:


t= np.arange(1,43)


# In[117]:


df1['t'] = t


# In[118]:


df1['t_sq'] = df1['t']*df1['t']


# In[119]:


log_Sales=np.log(df1['Sales'])


# In[120]:


df1['log_Sales']=log_Sales


# In[121]:


train1, test1 = np.split(df1, [int(.67 *len(df1))])


# In[122]:


linear= smf.ols('Sales ~ t',data=train1).fit()
predlin=pd.Series(linear.predict(pd.DataFrame(test1['t'])))
rmselin=np.sqrt((np.mean(np.array(test1['Sales'])-np.array(predlin))**2))
rmselin


# In[123]:


quad=smf.ols('Sales~t+t_sq',data=train1).fit()
predquad=pd.Series(quad.predict(pd.DataFrame(test1[['t','t_sq']])))
rmsequad=np.sqrt(np.mean((np.array(test1['Sales'])-np.array(predquad))**2))
rmsequad


# In[124]:


expo=smf.ols('log_Sales~t',data=train1).fit()
predexp=pd.Series(expo.predict(pd.DataFrame(test1['t'])))
rmseexpo=np.sqrt(np.mean((np.array(test1['Sales'])-np.array(np.exp(predexp)))**2))
rmseexpo


# In[125]:


output = {'Model':pd.Series(['rmselin','rmsequad','rmseexpo']),
          'Values':pd.Series([rmselin,rmsequad,rmseexpo])}


# In[126]:


rmse=pd.DataFrame(output)


# In[127]:


rmse


# In[ ]:




