import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,[3,4]].values ###? Here we will not take x and y seperately because this is a Unsuperwised Training Algorithm  , and here we are not predicting y with help of x but we are feeding all relevant information to the model and based on this it will make the required clusters

from sklearn.cluster import KMeans
wcss=[]##?Within-Cluster Sum of Squares (WCSS) is a crucial metric in K-Means clustering, representing the sum of squared distances between each data point and its corresponding cluster centroid. It helps in assessing the compactness of clusters, with lower WCSS values indicating more compact clusters.

for i in range(1,11):
    kmeans=KMeans(init='k-means++',n_clusters=i,random_state=45) ##? here we are initializing the clustering model inside the loop because this model be creadted again and again for every cluster , that's why we have assigned i in the poarameter of n_clusters , The K-means ++  makes sure that we do not run into the probem of random ititialisations and in randomstate we can write any number , That does not have a major significance on output 
    ##/The Elbow Method is a technique used to determine the optimal number of clusters (K) in K-Means clustering. It involves plotting the WCSS values for different K values and identifying the "elbow point" where the rate of decrease in WCSS slows down
    ##/ABOUT_ELBOW_METHOD:---In K-Means clustering, we start by randomly initializing k clusters and iteratively adjusting these clusters until they stabilize at an equilibrium point. However, before we can do this, we need to decide how many clusters (k) we should use.The Elbow Method helps us find this optimal k value. Here’s how it works:We iterate over a range of k values, typically from 1 to n (where n is a hyper-parameter you choose).For each k, we calculate the Within-Cluster Sum of Squares (WCSS).
    kmeans.fit(x)
    wcss.append(kmeans.inertia_) ##? This will append to the wcss list 
