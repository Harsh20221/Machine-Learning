import numpy as np
import tensorflow as tf
import pandas as pd
tf._version__
dataset=pd.read_csv('Churn_Modelling.csv')###? Accessing the Dartaset
x=dataset.iloc[:,3:-1].values#?Take column from credit score to estimated Salary
y=dataset.iloc[:,-1].values
###* ONE HOT ENCODING
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough') ##?Applying One Hot encoding on Country  Column
x=np.array(ct.fit_transform(x))
##* LABEL ENCODING
from sklearn.preprocessing import LabelEncoder###? This will be used to perform encoding(Convert to numbers) to the Male/Female Column 
le=LabelEncoder()
x[:,2]=le.fit_transform(x[:,2])
 ###*SPLIT INTIO TRAINING AND TEST SET
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=0)
##* FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test) ###?When sc.transform(x_test) is called, the test data is scaled using the same mean and standard deviation that were computed from the training set. This ensures that the test data is transformed in the same way as the training data.






