#project on stock market analysis using python and machine learning...
#to predict stock of(TSLA(NASDAQ)) USING PAST 60 DAYS OF STOCK PRICE....

#IMPORT THE LIBRARIES....
import numpy as np
import pandas as pd
import seaborn as sb
import datetime as dt
import plotly.graph_objects as go
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout  
from keras.models import Sequential
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
plt.style.use('bmh')
import warnings
warnings.filterwarnings('ignore')

#load the data
from google.colab import files
uploaded=files.upload()
#upload data
df=pd.read_csv('TSLAE.csv')
df.head(60)

#get the nuber of trading days
df.shape

df.info()

#visulaze the close price data
plt.figure(figsize=(16,8))
plt.title('TESLA')
plt.xlabel('DAYS')
plt.ylabel('Close price USD($)')
plt.plot(df['Close'])
plt.show()

#get the close price
df=df[['Close']]
df.head(10)

#Now let’s draw the distribution 
#plot for the continuous features given in the dataset.
#it implies no null values in the data set...
df.isnull().sum()

#create a variable to predict'x' days out into the future
future_days=25
#create a new column(target) shifted 'x' units/days up
df['Prediction']=df[['Close']].shift(-future_days)
df.head(4)

#create a variable to predict'x' days out into the future
future_days=25
#create a new column(target) shifted 'x' units/days up
df['Prediction']=df[['Close']].shift(-future_days)
df.tail(4)

#create the feature data set(x) and 
#convert it to a numpy array and remove the last x row/days
X=np.array(df.drop(['Prediction'],1)) [:-future_days]
print(X)

#create the target data set(y) and 
#convert it to a numpy aaray and get all of thetrarget value except the last"x" rows
y=np.array(df['Prediction'])[:-future_days]
print(y)

#split the data into 75% training and 25% testing
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

#Create the models
#create the decision tree regressor model
tree=DecisionTreeRegressor().fit(x_train,y_train)
#create linear regression model
lr=LinearRegression().fit(x_train,y_train)

#get the last 'x' rows of the figure data set
x_future =df.drop(['Prediction'],1)[:-future_days]
x_future=x_future.tail(future_days)
x_future=np.array(x_future)
x_future

#show the model tree prediction
tree_prediction=tree.predict(x_future)
print(tree_prediction)
print()
#show the model linear regression prediction
lr_prediction=lr.predict(x_future)
print(lr_prediction)

#visualize the data...
predictions=tree_prediction
valid=df[X.shape[0]:]
valid['Prediction']=predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close Price($)')
plt.plot(df['Close'])
plt.plot(valid[['Close','Prediction']])
plt.legend(['org','val','Pred'])
plt.show()

#visualize the data...
predictions=lr_prediction
valid=df[X.shape[0]:]
valid['Prediction']=predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close Price($)')
plt.plot(df['Close'])
plt.plot(valid[['Close','Prediction']])
plt.legend(['org','val','Pred'])
plt.show()
#data uploaded
df=pd.read_csv('TSLAE.csv')
df.head(60)

