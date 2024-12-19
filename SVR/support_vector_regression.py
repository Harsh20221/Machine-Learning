
###*Importing Libraries
import pandas as pd
import matplotlib as plt 
import numpy as np 
 ##* Importing the dataset and reshaping the values because Feature scvaling needs a 2d Array so we need to do this 
dataset=pd.read_csv('Position_Salaries.csv')##!!IF YOU get CSV Read Error then please OPEN THE core folder where you are coding like for this open the svr folder instead of opening the Machine learning folder as that can cause read csv errors
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
y=y.reshape(len(y),1)
print(y.reshape(len(y),1))
##* Implimenmting Feature Scaling on Both x and Y seperately 
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x=sc_x.fit_transform(x)
y=sc_y.fit_transform(y)

 