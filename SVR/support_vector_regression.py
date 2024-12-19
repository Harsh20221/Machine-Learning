
###*Importing Libraries
import pandas as pd
import matplotlib as plt 
import numpy as np 
 ##* Importing the dataset and reshaping the values because Feature scvaling needs a 2d Array so we need to do this 
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
y=y.reshape(len(y),1)
print(y.reshape(len(y),1))
##* Implimenmting Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler
x= sc.fit_transform
 