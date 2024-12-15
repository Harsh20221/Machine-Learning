import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
dataset= pd.read_csv('Data.csv') ##iloc is used to get the indexes 
x =  dataset.iloc[:,:-1].values ##This x specify matrix of feaatures ( All columns except the rightmost )
y= dataset.iloc[:,-1].values        ##This y is dependent variable vector containing only the last column
## in iloc bracket before comma specify rows and after comma we specify colums range or index               

##* In This Section We  HANDLE Missing Values by usually taking mean as our default approach in place of missing values 
from sklearn.impute import SimpleImputer ##/ This sklearn impute library is used for filling missing values  
print("-------------------------------------------" )
imputer = SimpleImputer(missing_values=np.nan,strategy='mean') 
imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])

 ###* HERE IN THE SECTION BELOW WE ARE ENCODING THE DATA MEANS WE ARE CONVERTING THE NON NUMERICAL VALUES OF OUR DATA LIKE NAME OF PLACE 
 ##* OR YES/NO TO NUMERICAL 1/0 , BINARY 001 OR 010 ETC  SO THAT OUR MACHINE LEARNING MODEL CAN PROCESS THOSE 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder=OneHotEncoder() #! Make sure you define one hot encoder here because column transformer expects onehotencoder as an instance not the class itself 
ct = ColumnTransformer(transformers=[('encoder',one_hot_encoder,[0])],remainder='passthrough') #?This Passthrough is used to keep the other remaning columns intact
#* after applying encoding to a column 
x=np.array(ct.fit_transform(x))
print(x)
 ###Transforming label to numericl values ( yes/no in dataset to 1 and 0)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y= le.fit_transform(y)  
print(y)


###*** HERE WE ARE SPLITTING OUR DATAset TO TRAINING SET AND TEST SET USING SKLEARN LIBRARY 
from sklearn.model_selection import train_test_split ###/ This Library will be used to handle splitting 
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=1) #?In this example, setting random_state=1 
#?ensures that the split between training and testing sets remains the same every time the script is run.




##* Doing Feature scaling on Whole Dataset  after seperating 
##what is feature scaling ?? --- feature scaling like leveling a playing field. 
##If one side of the field is much taller than the other, it’s hard to compare or compete fairly. Similarly, 
## scaling ensures all features are on a similar level, allowing machine learning models to process them more effectively.

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train[:,3:]=sc.fit_transform(x_train[:,3:])
x_test[:,3:]=sc.transform(x_test[:,3:])##? Use transform for test set as we want same scaler for test set as training set so instead of getting a new scaler using fit transform we use just transform for same scaler  




















""" print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
# Load the dataset
dataset=pd.read_csv('pima-indians-diabetes.csv')
# Identify missing data (assumes that missing data is represented as NaN)
missing_data=dataset.isnull().sum()
# Print the number of missing entries in each column
print(missing_data)
print ("------------------------------------------------")
# Configure an instance of the SimpleImputer class
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
# Fit the imputer on the DataFrame
imputer.fit(dataset)  #????Instead of specifying specific columns for fit you can just directly use it on the whole dataset and it will work
# Apply the transform to the DataFrame
dataset_changed=imputer.transform(dataset)##?? Make sure you print the new changed dataset to get the new results 
#Print your updated matrix of features
print(dataset_changed) """