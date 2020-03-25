import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

#Getting stock data
df = pd.read_csv("Google_StockTrain.csv")
df = df[['Open']]
#print(df)
#Variable for Predicting Some days
cast_out = 30

#Another Colomn
df['Prediction'] = df[['Open']].shift(-cast_out)
print(df)

#Creating new dataset before 30days
x = np.array(df.drop(['Prediction'],1))
x = x[:-cast_out]

y = np.array(df['Prediction'])
y = y[:-cast_out]

#Splitting and Training the data into 80 train and 20 testing
x_training, x_testing, y_training, y_testing = train_test_split(x,y,test_size=0.2)

#Values to be predicted
x_cast = np.array(df.drop(['Prediction'],1))[-cast_out:]
print(x_cast)

#Create and traing the SVM
svr_rbf = SVR(kernel='rbf',C=1e3,gamma=0.1)
svr_rbf.fit(x_training,y_training)
#testing this svm model
svm_conf = svr_rbf.score(x_testing,y_testing)
print(svm_conf)

#Linear Regession model

lr = LinearRegression()
lr.fit(x_testing,y_testing)
#Testing Linear Regression model
lr_conf = lr.score(x_testing,y_testing)
print(lr_conf)

#Predicted values using LR
lr_predit = lr.predict(x_cast)
print("Using Linear Regression :",lr_predit)


#Predicted values using SVM
svm_predict = svr_rbf.predict(x_cast)
print("using Support Vector Machine :",svm_predict)