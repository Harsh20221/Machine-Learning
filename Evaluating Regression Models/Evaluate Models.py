
import  numpy as np
import matplotlib as plt
import pandas as pd 
#*READING CSV 
dataset=pd.read_csv("/Users/harshkumar/Machine Learning/Multiple Regression/50_Startups.csv")
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
np.set_printoptions(precision=2) 
""" print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)) """

###########HERE WE ARE TESTING THE SUITABILITY OF OUR REGRESSION MOODEL BASED ON THE PRICIPLE OF R^2 ##########
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_pred)
print(r2)
##/ WE WILL GET THE R2 VALUE EQUAL TO 0.94 WHICH IS FAIRTLY GOOD