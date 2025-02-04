import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
dataset=pd.read_csv('Ads_CTR_Optimisation.csv')
N=10000
d=10
ads_selected=[]
no_of_selections=[0]*d
sum_of_rewards=[0]*d
total_reward=0
for n in range (0,N):
    ad=0
    max_upperbound=0
    for i in range(0,d):
        if(no_of_selections[i]>0): ####/ IF THE AD HAS BEEN SELECTED ALREADY
            average_reward=sum_of_rewards[i]/no_of_selections[i]
            delta_i=math.sqrt(3/2*math.log(n+1)/no_of_selections[i])
            upper_bound=average_reward+delta_i
            ###/IF THE AD HAS NOT BEEN SELECTED ALREADY 
        else:
            upper_bound=1e400
        if(upper_bound>max_upperbound):
                max_upperbound=upper_bound
                ad=i
        ads_selected.append(ad)
        no_of_selections[ad] += 1
        reward=dataset.values[n,ad] 
        sum_of_rewards[ad]=sum_of_rewards[ad]+reward
        total_reward=total_reward+reward               
    #####* Plotting the Final Graph ##############
plt.hist(ads_selected)
plt.title('Histogram of ads Selected ')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected ')
plt.show()
    