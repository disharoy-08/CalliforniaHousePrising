#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Lets load the California House Pricing Dataset

# In[3]:


from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()


# In[4]:


housing.keys()


# In[5]:


type(housing)


# In[6]:


#Let's check the description of the dataset
print(housing.DESCR)


# In[7]:


print(housing)


# In[14]:


print(housing.data)


# In[15]:


print(housing.target)


# In[16]:


print(housing.feature_names)


# # Prepare the Dataset

# In[8]:


dataset = pd.DataFrame(housing.data, columns= housing.feature_names)


# In[23]:


dataset.head()


# In[9]:


dataset['price']= housing.target


# In[25]:


dataset.head()


# In[26]:


dataset.info()


# In[27]:


## Summarizing the stats of the data
dataset.describe()


# # Check the missing values

# In[10]:


dataset.isnull().sum()


# In[30]:


### Exploratory Data Analysis
## Correlation
dataset.corr()


# In[11]:


import seaborn as sns
sns.pairplot(dataset)


# 
# # Analyzing The Correlated Features

# In[12]:


dataset.corr()


# In[13]:


plt.scatter(dataset['AveBedrms'], dataset['price'])
plt.xlabel('AveBedrms')
plt.ylabel('price')


# In[14]:


plt.scatter(dataset['MedInc'], dataset['price'])
plt.xlabel('MedInc')
plt.ylabel('price')


# In[15]:


import seaborn as sns
sns.regplot(x='MedInc', y='price', data=dataset)


# In[16]:


sns.regplot(x='Population', y='price', data= dataset)


# In[17]:


sns.regplot(x='Latitude', y='price', data= dataset)


# # Independent and Dependent features

# In[19]:


x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]


# In[20]:


x.head()


# In[21]:


y


# # Train Test Split

# In[22]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state= 42)


# In[23]:


x_train


# In[24]:


x_test


# # Standardize the dataset

# In[25]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[26]:


x_train = scaler.fit_transform(x_train)


# In[27]:


x_test = scaler.transform(x_test)


# In[28]:


x_train


# # Model Training

# In[29]:


from sklearn.linear_model import LinearRegression


# In[30]:


regression = LinearRegression()


# In[31]:


regression.fit(x_train, y_train)


# In[32]:


## Print the coefficients and the intercept

print(regression.coef_)


# In[33]:


print(regression.intercept_)


# In[34]:



## on which parameters the model has been trained
regression.get_params()


# In[35]:


### prediction with Test Data
reg_pred = regression.predict(x_test)


# In[36]:


reg_pred


# # Assumptions

# In[38]:


## Plot a scatter plot for the prediction
plt.scatter(y_test, reg_pred)


# In[39]:


## Residuals
residuals = y_test - reg_pred


# In[40]:


residuals


# In[41]:


## Plot this residuals

sns.displot(residuals, kind="kde")


# In[42]:


## Scatter plot with respect to prediction and residuals
## Uniform distribution
plt.scatter(reg_pred, residuals)


# In[44]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test, reg_pred))
print(mean_squared_error(y_test, reg_pred))
print(np.sqrt(mean_squared_error(y_test, reg_pred)))


# # R square and adjusted R square

#  Formula: R^2 = 1- SSR/SST
#  R^2 = Coefficient of determination, SSR = sum of squares of residuals, 
#  SST = Total sum of squares    

# In[46]:


from sklearn.metrics import r2_score
score = r2_score(y_test, reg_pred)
print(score)


#  Adjusted R2 = 1 – [(1-R2)*(n-1)/(n-k-1)]
# 
# where:
# 
# R2: The R2 of the model n: The number of observations k: The number of predictor variables

# In[47]:


# Display adjusted R-sqared
1- (1-score)*(len(y_test)-1)/(len(y_test) - x_test.shape[1] -1)


# # New Data Prediction

# In[49]:


housing.data[0].reshape(1, -1)


# In[50]:


## Transformation of New Data
scaler.transform(housing.data[0].reshape(1,-1))


# In[51]:


regression.predict(scaler.transform(housing.data[0].reshape(1, -1)))


# # Pickling The Model file for Deployment

# In[52]:


import pickle


# In[53]:


pickle.dump(regression, open('housepredmodel.pkl', 'wb'))


# In[54]:


pickled_model = pickle.load(open('housepredmodel.pkl', 'rb'))


# In[55]:


### Prediction
pickled_model.predict(scaler.transform(housing.data[0].reshape(1, -1)))


# In[ ]:




