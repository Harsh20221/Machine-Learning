
###/ Here we are trying to use classification in our dataset and to do that we will be using logistic regression model 

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
dataset=pd.read_csv('Social_Network_Ads.csv')
x=dataset.iloc[:,:-1].values##? This time we will also take the first column because we are also considering the age parameter of our dataset , so we will take every column except the last column in our x variable 
y=dataset.iloc[:,-1].values##? y will remain the same as before , just the last column 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)##/ Always fit transform is used for feature scaling 
x_test=sc.fit_transform(x_test)##! Make sure you apply feature scaling on x train and  x test since we have already split our data and on x only using same scaler because we are applying only on x , if we would have been also applying it on y then we would have taken seperate scaler for y too 
print(x_train)
