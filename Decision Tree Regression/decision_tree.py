import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:-1].values##?Means select every column except the last , select all rows 
y=dataset.iloc[ :,-1].values ##? Means Select Only the last column , select all rows 
##* RESGRESSION STARTS HERE 
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)
###*Making The Plot 
x_grid=np.arange(min(x),max(x),0.1)##?This line here helps us to ceate a grid of values that helps make a smoother curve 
x_grid=x_grid.reshape(len(x_grid),1)##? This line helps us to reshape the grid into a 2d array with one column , the reshape method is used to change the shape of the array without changing the data (Overall we can say that this line and the above line is responsible for smoothning the curve and are not mandatory )
plt.scatter(x,y,color='Red')##? Whis will make the scatter plot of real Values 
plt.plot(x_grid,regressor.predict(x_grid))
plt.title("Decision Tree Regresion")##?Provides the Title 
plt.xlabel("Position Level")
plt.ylabel("'Salary")
plt.show()
#/ HENCE we can see that the curve obtained is like a staircase and is not very precise because  decision tree resgression is usually used for multiple feature dataset and not single feature dataset that we have here  but the implementation we have here can be implimented for any other dataset 