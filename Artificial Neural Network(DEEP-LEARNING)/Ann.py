import numpy as np
import tensorflow as tf
import pandas as pd
tf.__version__
dataset=pd.read_csv('Churn_Modelling.csv')###? Accessing the Dartaset
x=dataset.iloc[:,3:-1].values#?Take column from credit score to estimated Salary
y=dataset.iloc[:,-1].values
##* LABEL ENCODING
from sklearn.preprocessing import LabelEncoder###? This will be used to perform encoding(Convert to numbers) to the Male/Female Column 
le=LabelEncoder()
x[:,2]=le.fit_transform(x[:,2])
###* ONE HOT ENCODING //Use This when we have Multiple Columns 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough') ##?Applying One Hot encoding on Country  Column
x=np.array(ct.fit_transform(x)) 
 ###*SPLIT INTIO TRAINING AND TEST SET
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=0)
##* FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test) ###?When sc.transform(x_test) is called, the test data is scaled using the same mean and standard deviation that were computed from the training set. This ensures that the test data is transformed in the same way as the training data.
###* Initialize the TensorFlow Model 
ann=tf.keras.models.Sequential()
##* Adding Primary Hidden Layer
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
#* Adding  Secondary hidden layer 
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
##* Adding Output Layer
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))##!!For Binary Classification we only use sigmoid activation but for non binary we use "softmax" classification

##* Applying the ANN  
#?In machine learning, optimizers and loss functions are two components that help improve the performance of the model. By calculating the difference between the expected and actual outputs of a model, metrics provide a list of metrics to be evaluated 
ann.compile(optimizer= "adam", loss ="binary_crossentropy" , metrics=["accuracy"])
#!!It is Important that whenever you are Doing Binary Classificatiojn you remember to use "binary_crossentropy" loss function only , For non binary it should be Categorical crossentropy

##* Training the Ann on the Training set
ann.fit(x_train,y_train,batch_size=32,epochs=100)

##*Predicting the Results for a Specific customer 
y_pred=ann.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]])) #??We are predicting for a particular customer whose nationalitry is France that's why we wrote one hot encoded value for France which is 1,0,0 found by printing it , and His credit score =600 , Gender=Male,Age=40,Tenure=3yrs,Balance=60000,No of products=2,Does the customer have a credit card=yes,If it's an Active customer=yes,Estimated Salary=50000,
##we are predicting if the customer will stay in the bank or leave 
#/!!!We Must wrap the diffrent parameters we wanna use for prediction with the scaler because these input values for prediction must be feature scaled and we use sc. for this , For This Feature Scaling here inside the predict we must not use the usual Predict Method or else it can lead to information leakage , We must use the transform Method Instead
print(y_pred>0.5) ##?The Output Will be Predicted probability in the form of 0.? if he'll stay or leave the bank and to convert this to a Binary result in the form of yes/no we'll use the following method >0.5 , It means if the predicted probability that the customer leaves the bank is below 0.5 then we consider it to be False and if the predicted value is above 0.5 then we consider it to be 1

###*Predicting for All Customers
Y_pred=ann.predict(x_test)
Y_pred=(Y_pred>0.5)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1), y_test.reshape(len(y_test),1)),1))##?This line will compare the predicted results to the actual results , The predicted results will be on the left and actual results will be on the right 





