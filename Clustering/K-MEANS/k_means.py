import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,[3,4]].values ###? Here we will not take x and y seperately because this is a Unsuperwised Training Algorithm  , and here we are not predicting y with help of x but we are feeding all relevant information to the model and based on this it will make the required clusters,    Also Train Test Split is not needed 
## x contains annual income and spending score of each customers 
####*Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss=[]##?Within-Cluster Sum of Squares (WCSS) is a crucial metric in K-Means clustering, representing the sum of squared distances between each data point and its corresponding cluster centroid. It helps in assessing the compactness of clusters, with lower WCSS values indicating more compact clusters.
for i in range(1,11):
    kmeans=KMeans(init='k-means++',n_clusters=i,random_state=45) ##? here we are initializing the clustering model inside the loop because this model be creadted again and again for every cluster , that's why we have assigned i in the poarameter of n_clusters , The K-means ++  makes sure that we do not run into the probem of random ititialisations and in randomstate we can write any number , That does not have a major significance on output 
    ##/The Elbow Method is a technique used to determine the optimal number of clusters (K) in K-Means clustering. It involves plotting the WCSS values for different K values and identifying the "elbow point" where the rate of decrease in WCSS slows down
    ##/ABOUT_ELBOW_METHOD:---In K-Means clustering, we start by randomly initializing k clusters and iteratively adjusting these clusters until they stabilize at an equilibrium point. However, before we can do this, we need to decide how many clusters (k) we should use.The Elbow Method helps us find this optimal k value. Here’s how it works:We iterate over a range of k values, typically from 1 to n (where n is a hyper-parameter you choose).For each k, we calculate the Within-Cluster Sum of Squares (WCSS).
    kmeans.fit(x)
    wcss.append(kmeans.inertia_) ##? This will append to the wcss list 
    ##*Plotting OPTIMAL NO OF CLUSTERS USING ELBOW METHOD GRAPH 
plt.plot(range(1,11),wcss)
plt.title("THE ELBOW METHOD")
plt.xlabel("No of Clusters")
plt.ylabel("WCSS")
plt.show()

##*Predicting the Actual Clusters where each group of customers will belong 
kmeans=KMeans(init='k-means++',n_clusters=5,random_state=45) ###!!!THIS K- MEANS WILL BE DEFINED AGAIN THIS TIME AND THIS TIME WE'LL WRITE THE NO OF CLUSTERS THAT WE WANNA MAKE IN PLACE OF n_clusters , we have taken 5 here in this case as an example 
y_kmeans=kmeans.fit_predict(x)
print(y_kmeans) ##/ THIS WILL PRINT THE NAME OF THE CLUSTERS ALONG WITH THE CUTOMERS WHO ARE ASSIGNED TO THAT CLUSTERS 
##*Creating ScatterPlot for all clusters 
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c='red',label="Cluster-1") ##? HERE WE ARE CREATING SCATTER PLOT FOR EVERY CLUSTER ( DISPLAYED ALLTOGETHER IN ONE GRAPH), IN this step what we mean by x[y_kmeans==0,0] & x[y_kmeans==[0,1] is first 0 in both of them define the cluster number that is why we put zero in both of them, as this is for cluster no zero then after comma we again  wrote 0 in first and this time we wrote one in second because the first zero is for  the first feature of the dataset that is annual income and in second place one  is for the spending score , This is how we'll write for other clusters too , To check the names of diferent clusters use the above print(y_kmeans) print statement 
##? by s we mean the size of the points in the scatter plot , c is for the color of the points , label is for the name of the cluster
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c='blue',label="Cluster-2")
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c='green',label="Cluster-3")
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,c='cyan',label="Cluster-4")
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=100,c='magenta',label="Cluster-5")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label="Centroids") ##? This will plot the centroids of the clusters
plt.title("Clusters of Customers")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()##? This will show the labels of the clusters, describes which color is for which cluster
plt.show()

