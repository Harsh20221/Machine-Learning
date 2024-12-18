#importing libraries
import  numpy as np
import matplotlib as plt
import pandas as pd 
#*READING CSV 
dataset = pd.read_csv(r'C:\MAchine Learning\Machine-Learning\Multiple Regression\50_Startups.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
#*ENCODING -- CONVERTING EVERY COLUMN TO NUMERALS 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
x=np.array(ct.fit_transform(x))
#* Seperating to Test and training Set 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#*Applying Regresion 
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
#* Predicting Results 
y_pred=regressor.predict(x_test)
np.set_printoptions(precision=2) ##? This line configures NumPy's print settings to display floating-point numbers with a precision of two decimal places. 
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))##?This line performs several operations to align and combine the predicted values (y_pred) with the actual test values (y_test) for comparison:

##/FAQ -- y_pred.reshape(len(y_pred), 1): Transforms the y_pred array, which likely has a shape of (n,), into a two-dimensional array with shape (n, 1). This ensures that the predicted values are organized as a column vector.

##/ --- y_test.reshape(len(y_test), 1): Similarly, reshapes the y_test array into a two-dimensional column vector with shape (n, 1).

##/----Reshaping is crucial because it standardizes the dimensions of both arrays, making them compatible for concatenation. Without reshaping, attempting to concatenate one-dimensional arrays along a specified axis could lead to unexpected results or errors.
##/The overall effect of this concatenation is to create a combined array where each row presents a pair of predicted and actual values. This side-by-side arrangement is particularly useful for evaluating the performance of the regression model, as it allows for easy visual or programmatic comparison between what the model has predicted and the true values from the test set.

