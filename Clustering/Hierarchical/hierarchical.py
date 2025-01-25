import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,[3,4]].values
import scipy.cluster.hierarchy as sch
dendogram=sch.dendrogram(sch.linkage(x,method='ward')) ##/ The ward method refers to the minimum variance method
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidian Distance')
plt.show()
##*Predicting the clusters
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5,  linkage = 'ward')###!!!DO NOT ADD Affinity parameter in the AgglomerativeClustering because adding it will generate errors
y_hc=hc.fit_predict(x)
print(y_hc)
##*Making the cluster plot
plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=100,c='red',label='Cluster-1') ##?s stands for size while c stands for color  and label for name 
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=100,c='blue',label='Cluster-2')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.xlabel('Annual-Income')
plt.ylabel('Spending-Score')
plt.legend()##?To add legends to our graph (not mandatory)
plt.show()





