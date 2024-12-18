import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt ##!Always import matplotlib.pypot and do not just import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
dataset=pd.read_csv("/Users/harshkumar/Machine Learning/Polynomial Regression/Position_Salaries.csv")
x=dataset.iloc[:,1:-1].values#? We wanna take all the columns from the second , we do not wanna include the company position columns
y=dataset.iloc[:,-1].values 

#!!HERE IN THIS POLYNOMIAL REGRESSION EXAMPLE WE WILL NOT SPLIT THE DATA INTO TRAINING SET AND TEST SET BECAUSE WE WANT TO UTILISE THE WHOLE SET AS MUCH AS POSSIBLE FOR US 
""" from sklearn.model_selection import train_test_split
x_train ,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
"""

#*Training Linear Regression Model 
lin_reg= LinearRegression()
lin_reg.fit(x,y)

#* Training Polynomial regression Model on the whole dataset 
poly_reg=PolynomialFeatures(degree=6)##? Means we want a polynomial regressor with 2 matrix of features
##/ Here  as we increase the degree more from 2 to even higher then we will have much more accurate results , Our predictions will be more aligned with the actual results , For ex- chossing degree 6 will almost align the regression curve(Predicted values Curve ) in a picture perfect way with the actual scatter plot containing actual values 
X_poly=poly_reg.fit_transform(x) ##? This creates the feature matrix of polynomial regression(The resulting feature matrix with added polynomial terms, enabling the regression model to capture nonlinear relationships.) 
#* Creating a new Linear Regression Model out of the values recieved from the polynimial regressor 
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)#? Here we are creating a new linear regression model by taking x from the newly made polynomial regressor fit transform 


#*Plotting the Linear Regression Model
plt.figure("Linear Regression")##!!VERY IMPORTANT TO MENTION FIGURE IF YOU WANT TO OPEN BOTH PLOTS (LINEAR AND POLYNOMIAL REGRESSION) IN SEPERATE WINDOWS 
plt.scatter(x,y,color='Red') ## Here using Scatter function of the matplot we display the actual position level vs salary and mark it using scatter points ( Red in color ) , These are actual values 
plt.plot(x,lin_reg.predict(x),color='Blue') ###Here we are creating the plot of the Linear regression model that we defined (This is not linked with polynomial regresion plt , we are  making it for referance ), the first argument of the plot functin the x axis is our position level that we know from the dataset and in the second argument we are feeding that position level to the regression model to predct and display predicted salary in the y axis ,(THIS  CONTAINS  PREDICTED VALUES )   
plt.title("SALARY PREDICTION(Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")

#/ HENCE WE CAN SEE  THAT BY JUST USING THE LINEAR REGRESSION MODEL FOR THIS DATASET THE PREDICTIONS ARE VERY FAR FROM THE ACTUAL VALUES IN THE GRAPH , HENCE WE CAN CONCLUDE THAT LINEAR REGRESSION IS NOT  SUITED FOR DATSETS  LIKE THIS  

##* Plotting the Polynomial Regression  Model
plt.figure("Polynomial Regression")##!!VERY IMPORTANT TO MENTION FIGURE IF YOU WANT TO OPEN BOTH PLOTS (LINEAR AND POLYNOMIAL REGRESSION) IN SEPERATE WINDOWS 
plt.scatter(x,y,color='Red') ##? The scatter function will remain the same as we are displaying the actual data and hence this will also remain same in our polynomial regression model 
plt.plot(x,lin_reg2.predict(X_poly),color='Blue') ### Here we are predicting and showing results according to the polynomial regression , the x axis will remain x  as like the linear regression but for y axis where we are making predictions will be different because in y axis we will enter polynomial regressor variable  instead of just entering x to make prediction 
plt.title("SALARY PREDICTION (Polynomial Regression)")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

###/We can see that the polynomial Regression  Plot is much more aligned with the actual values of the datasheet and the predictions are  more accurate then  by just using linear regression 



