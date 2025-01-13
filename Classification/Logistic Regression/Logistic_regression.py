
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
###* Applying our classification model 
from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)##? Here we are fitting the model into our dataset
##* Predicting for a single value 
classifier.predict(sc.transform([[30,87000]])) ##!!Make sure you always uise sc.transform when you are making prediction on feature scaled data
##!! Make sure you write the prediction parameters inside a 2d array if you have more than one parameter for prediction 
##? In the above code snippet we are predicting fore someone with age=30 and salary = 87000 whether he/she will buy the car or not 
####* Predicting the Test Set Results 
y_predict=classifier.predict(x_test)
""" print(y_predict) """ ##? This will print the prediction 
print(np.concatenate( (y_predict. reshape (len(y_predict) ,1), y_test.reshape(len(y_test) ,1)), 1))##?The active selection is a Python statement that prints the result of concatenating two arrays, y_predict and y_test, along the second axis (columns). This is achieved using the np.concatenate function from the NumPy library, which joins a sequence of arrays along an existing axis.                                                    
##?First, both y_predict and y_test arrays are reshaped to be two-dimensional with a single column. The reshape method is used for this purpose, where len(y_predict) and len(y_test) specify the number of rows, and 1 specifies the single column. This transformation ensures that both arrays have compatible shapes for concatenation along the specified axis
###In the Given output of the above following statement the first column will show the prediction of the model while the next column will show  the actual result 
###* Making the confusion Matrix  to find the accuracy of the model 
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predict)###!!!!!!Make sure you do not take x_test here or eklse it will not work because we are predicting for y 
print(cm)
