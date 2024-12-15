import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##* Importing the dataset
dataset = pd.read_csv("Salary_Data.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
##* Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=0) 
##*  Using the Linear Regression model to train the model And Accessing Model Accuracy 
from sklearn.linear_model import LinearRegression ##Importing the LinearRegression class from the linear_model module of the sklearn library
regressor=LinearRegression() ##Creating an object of LinearRegression class
regressor.fit(x_train,y_train) ##Here we are putting our training data to the regression model and it will learn the correlation between the data , the fit function will train the model
y_pred = regressor.predict(np.array(x_test)) ### Here we are generating the predicted values of the test data based on the model we trained
##/NOTE-- Predictions (y_pred) are made using x_test to compare against y_test for assessing accuracy.
##* Visualising the Training set results

plt.figure("Training Set")
plt.scatter(x_train,y_train,color='red')##?Plots the actual training data points in red.
plt.plot(x_train,regressor.predict(x_train),color='blue')##?Draws the regression line based on the training data.
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')


###/NOTE--Why Not Use x_test for Plotting the Regression Line?
###/Reason:Regression Line is Based on Training Data:The regression line is derived from the training data. 
#/It represents the model's understanding of the relationship between features and target based on what it learned during training. 
#/Using x_test for plotting the regression line would not make sense because the line doesn't "know" the test data; it's solely based on the training data

##* Visualising on Real  Set 

plt.figure("Test Set")
plt.scatter(x_test,y_test,color='red')###?Plots the actual test data points in red.
plt.plot(x_train,regressor.predict(x_train),color='blue')###?Same Regression Line: Uses the regression line derived from the training data.
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()##!!!!! WRITE plt.show() only once at the end if you wanna show both plots simoultanously if you write it twice or not at the last position then both plots will not display simoultanously 

##/SUMMARY OF CONFUSION BETWEEEN USING X_test / X_Train 
##/ x_train and y_train:
#/ Used to train the regression model.
#/Helps in deriving the regression line that represents the learned relationship.
#/Visualized alongside the regression line to assess the fit on training data.
 
#/x_test and y_test:
#/Used to evaluate the model's performance on unseen data.
#/Plotted as scatter points to compare actual outcomes with the trained model's predictions.
#/Helps in understanding the model's generalization ability.


