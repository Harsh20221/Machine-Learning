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
