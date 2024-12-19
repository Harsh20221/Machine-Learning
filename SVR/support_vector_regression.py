
###*Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt  ##!! ALWAYS IMPORT PYPLOT FROM matplotlib or else youn will face a nasty error , Please do not forget or this will lead to a very silly mistake but annoyng one 
import numpy as np 
 ##* Importing the dataset and reshaping the values because Feature scvaling needs a 2d Array so we need to do this 
dataset=pd.read_csv('Position_Salaries.csv')##!!IF YOU get CSV Read Error then please OPEN THE core folder where you are coding like for this open the svr folder instead of opening the Machine learning folder as that can cause read csv errors
x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values
y=y.reshape(len(y),1)

##* Implimenmting Feature Scaling on Both x and Y seperately , We are doing feature scaling so that the values are not too far from each other 
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
X=sc_x.fit_transform(x) ###!! VERY IMPORTANT TO ASSIGN THE NOW Fit transformed data to a new variable else if you assign it to the old x then  it will create errors 
Y=sc_y.fit_transform(y)
print(X)
print(Y)


###* Initialising the SR Regression Model
from sklearn.svm import SVR
regressor= SVR(kernel='rbf') ##? Make sure to Select the kernel in the svm ,Rbf means radial  basis function
regressor.fit(X,Y) ##? This fit will begin the regression 

##* Reversing the scaling -- Because we have applied feature scaling on both x and y so the prediction model will generate the results around the same scale but now we wanna get the results in the actual scale whose value we did enter so now we will do reverse feature scaling 
sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1,1)) ##? Unlike polynomial regression where we used fit transform to just get prediction for 6.5 we will use transform here to get the predictio n and at the same time we will descale it into the original scale using inverse transform

##** Ploting the Regression Curve 
plt.scatter(sc_x.inverse_transform(X),sc_y.inverse_transform(Y) ,color='RED')
plt.plot(sc_x.inverse_transform(X),sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)),color="BLUE")
plt.title("SVR Regression Curve  ")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
