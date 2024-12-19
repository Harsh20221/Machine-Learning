
###*Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt  ##!!!!!!!!!!!! ALWAYS IMPORT PYPLOT FROM matplotlib or else youn will face a nasty error , Please do not forget or this will lead to a very silly mistake but annoyng one 
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
plt.figure("SVR Regression ")##? The figure is used to Display both graphs simoultanously 
plt.scatter(sc_x.inverse_transform(X),sc_y.inverse_transform(Y) ,color='RED') ##!! Make sure to input the New Feature scaled values of  x and y here  then descale them here itself to generate scatter plots 
##/ Why Raw values of x and y is not used in the scatter plot???????
#/ While it's possible to use the raw data x and y directly for plotting the scatter points, the code uses sc_x.inverse_transform(X) and sc_y.inverse_transform(Y) 
#/ because: 
#/ The original x and y may no longer contain the raw data due to variable overwriting or transformations.It ensures consistency with the rest of the code that operates on the scaled data.It avoids potential errors and keeps the data handling streamlined. By inverse-transforming the scaled data for plotting, you retrieve the original values of x and y, making the scatter plot accurately reflect the actual data, even after scaling has been applied for modeling purposes.
plt.plot(sc_x.inverse_transform(X),sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)),color="BLUE")##! Make sure to do not forget to add the reshape here else error will persist in plotting  and similarly input the feature scaled values of x and y and descale them here only 
plt.title("SVR Regression Curve ")
plt.xlabel("Position Level")
plt.ylabel("Salary")



##* Making a More Precise Curve 
plt.figure("SVR Regression Curve Precise")
X_grid = np.arange(min(sc_x.inverse_transform(X)), max(sc_x.inverse_transform(Y)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_x.inverse_transform(X),sc_y.inverse_transform(Y) ,color='RED')
plt.plot(X_grid,sc_y.inverse_transform(regressor.predict(sc_x.transform(X_grid)).reshape(-1,1)),color="BLUE") ##? WE USED THE sc_x.transform again inside the predict function of the resgressor predict because here we are using X_grid which was not feature scaled so first we will feature scale it then again descale it 
plt.title("SVR Regression Curve Pecise ")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
