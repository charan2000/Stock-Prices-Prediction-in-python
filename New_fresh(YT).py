# -*- coding: utf-8 -*-
"""
Modified on Mon Mar 30 00:33:10 2020

@author: Charan
"""

import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Method for Data_required 
def Data_graph():       
    df = pd.read_csv('Book1.csv')
    #Getting required data for graph
    dates = df.loc[:, 'Date']
    Open = df.loc[:, 'Open']

    #Create the independent data set 'X'
    Date = []
    for date in dates:
        Date.append([int(date.split('/')[1])])

    #Create the dependent data set 'y'
    Prices = [] 
    for open_price in Open:
        Prices.append(float(open_price))
    return Date,Prices
    
def graph(dates, prices, x):
    #Create the 3 Support Vector Regression models
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
  
    #Train the SVR models 
    svr_rbf.fit(dates,prices)
  
    #Create the Linear Regression model
    lin_reg = LinearRegression()
    #Train the Linear Regression model
    lin_reg.fit(dates,prices)
  
    #Plot the models on a graph to see which has the best fit
    print("This graph is to show the Support Vector Regressor:")
    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, lin_reg.predict(dates), color='yellow', label='Linear Reg')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='SVR RBF')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title('Regression')
    plt.legend()
    plt.show()
    #return svr_rbf.predict(x)[0]

#Getting stock data
df = pd.read_csv("Google_StockTrain.csv")
df = df[['Open']]

#Variable for Predicting Some days
cast_out = 30

#Another Colomn
df['Prediction'] = df[['Open']].shift(-cast_out)

#Creating new dataset before 30days for testing!
x = np.array(df.drop(['Prediction'],1))
x = x[:-cast_out]
y = np.array(df['Prediction'])
y = y[:-cast_out]

#Splitting and Training the data into 80 train and 20 testing
x_training, x_testing, y_training, y_testing = train_test_split(x,y,test_size=0.2)

#Values to be predicted
x_cast = np.array(df.drop(['Prediction'],1))[-cast_out:]
print("Data we are going to use:",x_cast)

#Create and traing the SVM
svr_rbf = SVR(kernel='rbf',C=1e3,gamma=0.1)
svr_rbf.fit(x_training, y_training)

#testing this svm model
svm_conf = svr_rbf.score(x_testing, y_testing)
print(svm_conf)

#Linear Regession model \
lr = LinearRegression()
lr.fit(x_testing, y_testing)

#Testing Linear Regression model
lr_conf = lr.score(x_testing, y_testing)
print(lr_conf)

#Predicted values using LR
lr_predit = [lr.predict(x_cast)]
print("Using Linear Regression :",lr_predit)

svr_predict = [svr_rbf.predict(x_cast)]

#Predicted values using SVM
print("Using SVR:",svr_predict)

dgraph = Data_graph()
Date,Prices = dgraph 

g = graph(Date,Prices,[[28]])
print(g)
