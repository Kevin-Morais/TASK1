# TASK1
# Task_1: Prediction using supervised Machine learning
### Name : Anto Waaltter Kevin Morais

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import numpy.ma.mrecords as mrecords



#Import the data
url="http://bit.ly/w-data"
data=pd.read_csv(url)
data1=data
print("The data is imported successfully")
data

data.describe()


# DATA VISUALIZATION


#Plotting the distribution of scores
data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

## Linear Regression Model


#Splitting training and testing data
x=data.iloc[:,:-1].values
y=data.iloc[:,1].values
x_train, x_test, y_train, y_test= train_test_split(x, y,train_size=0.80,test_size=0.20,random_state=0)


from sklearn.linear_model import LinearRegression
linearRegressor= LinearRegression()
linearRegressor.fit(x_train, y_train)
y_predict= linearRegressor.predict(x_train)


regressor = LinearRegression()  
regressor.fit(x_train, y_train) 

print("Training complete.")



# Plotting the regression line
line = regressor.coef_*x+regressor.intercept_
# Plotting for the test data
plt.scatter(x, y)
plt.plot(x, line);
plt.show()

print('Test Score')
print(regressor.score(x_test, y_test))
print('Training Score')
print(regressor.score(x_train, y_train))
y_pred=regressor.predict(x_test)

data= pd.DataFrame({'Actual': y_test,'Predicted': y_pred})
data

data.plot(cmap='Set1',figsize=(8,5))
plt.show()

print('Score of student who studied for 9.25 hours a day', regressor.predict([[9.25]]))


#Checking the efficiency of model
mean_squ_error = mean_squared_error(y_test, y_pred)
mean_abs_error = mean_absolute_error(y_test, y_pred)
print("Mean Squred Error:",mean_squ_error)
print("Mean absolute Error:",mean_abs_error)
