
## Code is almost similar to logistic regression , Just one Block is different
##(THE PLOT OF THIS METHOD IS COMPUTE INTENSIVE )
##* Initializing the Dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
dataset = pd.read_csv('Social_Network_Ads.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
####* Seperate the Data Into test and train set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)
###*Apply Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train) ##/ Always remember that in the Standard Scaler we scale the the x parameter , means we scale the training set and  test set that belongs to x only not y  because only our x is having large values , while the  y contains the values under the range so we only apply feature scaling on the data fields which are very large to normalise them and not the whole dataset 
x_test=sc.fit_transform(x_test)
##* Initialize the Classifier 
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)##?We choose the Minkowski metric because we want to choose the classic Euclidean distance system for this classification purpose  , we also input p=2 for the very same purpose
classifier.fit(x_train,y_train)##/ Also Remember that unlike the feature scaling we apply the fit method to the x train and y train both because these are the oparameters on which the prediction will be going to happen so we need to feed all these to the classifier , We do not  feed x_test or y_test in this fit method as  we will be evaluating our model based on the  test set  both x and y 
###** Predicting the Results of the Dataset 
y_pred=classifier.predict(x_test)###/Note that we always make the prediction based on the testset represented by x_test 
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))##? Printing the result of the prediction and reshaping the array such that the first column is the predictiopn column  having values 0 and 1 where 0 means no the customer will not buy the suv and the second  column is the column contains the actual values and from the both columns we can judge if the prediction became true or false 
#### if the print result is like 00 or 11 then it means our prediction is correct but if it is like 01  means the predeiction is wrong and customer did buy the car eventhough our model predicted that he/she will not buy the car 
####* Creating the Confusion matrix  and Accuracy Score 
from sklearn.metrics import confusion_matrix, accuracy_score##? Importing both libraries 
cm = confusion_matrix(y_test, y_pred) ###/ Always remember that confusion matrix will be created upon the predicted y and the y testset that we created 
print(cm)
accuracy_score(y_test, y_pred)##/ same parameters will be used for the accduracy score too 

##* Visualizing Training Set (Code not too important )
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(x_train), y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.5),
                     np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.5))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(['#FA8072', '#1E90FF']))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(['#FA8072', '#1E90FF'])(i), label=j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
##* Visualizimng Test Set (Code not too important )
X_set, y_set = sc.inverse_transform(x_test), y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.5),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.5))
Z = classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)
plt.contourf(X1, X2, Z, alpha=0.75, cmap=ListedColormap(['#FA8072', '#1E90FF']))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
colors = ['#FA8072', '#1E90FF']
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], color=colors[i], label=j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
