#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf


# In[2]:


company = yf.Ticker("SBIN.NS")


# In[ ]:


#start_date = input("Enter the starting date in yyyy/mm/dd format : ")
#end_date = input("Enter the end date : ")


# In[ ]:


## Start and Ending of the dataset


# In[3]:


dff = company.history(period="1y")


# In[4]:


dff.head()


# In[5]:


df = dff[['Open']]


# In[18]:


#Variable for Predicting Some days
cast_out = 10

#Another Colomn
df['Prediction'] = dff[['Open']].shift(-cast_out)
df


# In[19]:


#Creating new dataset before 30days for testing!
x = np.array(df.drop(['Prediction'],1))
x = x[:-cast_out]
y = np.array(df['Prediction'])
y = y[:-cast_out]


# In[20]:


#Splitting and Training the data into 80 train and 20 testing
x_training, x_testing, y_training, y_testing = train_test_split(x,y,test_size=0.33,random_state=52)


# In[21]:


#Values to be predicted
x_cast = np.array(df.drop(['Prediction'],1))[-cast_out:]
print("Data we are going to use:",x_cast)


# In[23]:


#Linear Regession model \
lr = LinearRegression()
lr.fit(x_testing, y_testing)

#Testing Linear Regression model
lr_conf = lr.score(x_testing, y_testing)
print(lr_conf)

#Predicted values using LR
lr_predit = [lr.predict(x_cast)]
print("Using Linear Regression :",lr_predit)


# In[ ]:


#Create and traing the SVM model
svr_rbf = SVR(kernel='rbf',C=1e3,gamma=0.1)
svr_rbf.fit(x_training, y_training)

#testing this svm model# ## Gives the score
svm_conf = svr_rbf.score(x_testing, y_testing)
print(svm_conf)

#Predicted values using SVM

svr_predict = [svr_rbf.predict(x_cast)]
print("Using SVR:",svr_predict)


# In[ ]:




