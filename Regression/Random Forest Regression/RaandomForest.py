import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt

dataset=pd.read_csv("Position_Salaries.csv")

x=dataset.iloc[:,1:-1].values ####? x will only include the levels 
y=dataset.iloc[:,-1].values##? This will only contains the salaries 
###* Initialise Regressor
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=10,random_state=0)##?This parameter specifies the number of trees in the forest. In this case, n_estimators=10 means that the random forest will consist of 10 decision trees. Increasing the number of trees can improve the model's performance but also increases computational cost.
##?random_state=0 ensures that the results are reproducible, meaning that running the code multiple times will produce the same results.
##?random_state: This parameter controls the randomness of the bootstrapping of the samples used when building trees. Setting
regressor.fit(x,y) ###fit() is a function that will fit an equation or model to data
###* Draw the plots 
X_grid = np.arange(min(x), max(x), 0.01) ##? This is to make the curve more smoother 
X_grid = X_grid.reshape((len(X_grid), 1))##? To make curve more smoother we also reshape 
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')##?Plotting the curve 
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()