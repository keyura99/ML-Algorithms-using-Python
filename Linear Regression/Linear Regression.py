#!/usr/bin/env python
# coding: utf-8

# # Build Linear Regression Model in Python

# In this Jupyter notebook, I will be showing you how to build a linear regression model in Python using the scikit-learn package on a Mobile Price Prediction Dataset from Kaggle

# In[1]:


#Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split


# In[2]:


#Load Dataset
data=pd.read_csv("C:/Users/vadla/Downloads/Cellphone.csv")


# In[3]:


#Quick View about Dataset
data


# In[4]:


data.shape # Number of rows and columns


# In[5]:


data.describe() #statistical Analysis


# In[6]:


#Data Cleansing:
data.info()


# All of the features of this dataset belongs to either "Int" or "Float" Type.
# There are No Null values in this dataset.

# ## Data Visualization

# We will be using Scatter plots. They will observe the relationship between variables and uses dots to represent the connection between them. 

# In[7]:


plt.scatter(data["weight"], data["Price"])


# In[8]:


plt.scatter(data["resoloution"], data["Price"])


# In[9]:


plt.scatter(data["ppi"], data["Price"])


# In[10]:


plt.scatter(data["battery"], data["Price"])


# In[11]:


plt.scatter(data["thickness"], data["Price"])


# In[12]:


plt.scatter(data["Front_Cam"], data["Price"])


# In[13]:


plt.scatter(data["RearCam"], data["Price"])


# In[14]:


plt.scatter(data["ram"], data["Price"])


# In[15]:


plt.scatter(data["cpu core"], data["Price"])


# In[16]:


plt.scatter(data["cpu freq"], data["Price"])


# In[17]:


plt.scatter(data["internal mem"], data["Price"])


# In[18]:


sns.pairplot(data)


# The pairplot function creates a grid of Axes such that each variable in data will by shared in the y-axis across a single row and in the x-axis across a single column. That creates plots as shown above

# From the visulizations,we can deduce that almost all the features have a linear relationship

# # Checking correlation between features

# In[19]:


data=data.drop(columns=["Product_id","Sale"]) #Dropping unwanted columns as they don't add any value to our analysis
corr=data.corr()
import plotly.express as px
fig = px.imshow(corr, text_auto=True,width=1000, height=1000)
fig.show()


# # Split datset into X and Y variables

# In[20]:


x=data.drop('Price',axis=1)
y=data['Price']


# In[21]:


x.shape


# In[22]:


y.shape


# # Perform 80/20 Data split

# Since I have already imported train test split, we can split the data into 80:20. Most commonly the ratio used to split the data is 80:20. This is done so that we or our model don't see a particular set of data and is kept aside for testing our trained model. And the larger set is always used for training and the latter for testing.

# In[23]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# # Checking data dimensions

# In[24]:


x_train.shape, y_train.shape


# In[25]:


x_test.shape, y_test.shape


# # Linear Regression Model

# In[26]:


#Importing necessarylibraries
#from sklearn import linear_model-already imported
from sklearn.metrics import mean_squared_error, r2_score


# # Build linear regression

# #### Defines the regression model

# In[27]:


model = linear_model.LinearRegression()


# #### Build training model

# In[28]:


model.fit(x_train, y_train)


# #### Apply trained model to make prediction (on test set)

# In[29]:


y_pred = model.predict(x_test)


# # Prediction results

# #### Print model performance

# In[30]:


print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(y_test, y_pred))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(y_test, y_pred))


# # String Formatting

# By default r2_score returns a floating number

# In[31]:


r2_score(y_test, y_pred).dtype


# In[32]:


r2_score(y_test, y_pred)


# We will be using the modulo operator to format the numbers by rounding it off.

# In[33]:


'%.2f' %0.9609474449922408


# the Result for R^2 is 0.96 that is very good and showing that Linear regression model is Right Choice.
# It means that independent variables describe 96% of dependent variable!

# In[ ]:




