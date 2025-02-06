import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
dataset=pd.read_csv('Ads_CTR_Optimisation.csv')
N=10000##?This line initializes N to 10000, representing the total number of rounds or iterations the algorithm will run.
d=10###?This line sets d to 10, indicating the number of different ads available.
ads_selected=[]
no_of_selections=[0]*d###?This line initializes a list no_of_selections with d elements, all set to 0. It keeps track of the number of times each ad has been selected.
sum_of_rewards=[0]*d###?This line initializes a list sum_of_rewards with d elements, all set to 0. It keeps track of the total reward accumulated for each ad.
total_reward=0##?This line initializes total_reward to 0, which will keep a running total of all rewards obtained.
for n in range (0,N):
    ad=0
    max_upperbound=0
    for i in range(0,d):
        if(no_of_selections[i]>0): ####/ IF THE AD HAS BEEN SELECTED ALREADY
            average_reward=sum_of_rewards[i]/no_of_selections[i]
            delta_i=math.sqrt(3/2*math.log(n+1)/no_of_selections[i])###?  This line calculates delta_i, which is the exploration term. It uses the square root of the logarithm of the current round number divided by the number of times the ad has been selected, scaled by a factor of 3/2.  
            upper_bound=average_reward+delta_i
            ###/IF THE AD HAS NOT BEEN SELECTED ALREADY 
        else:
            upper_bound=1e400
        if(upper_bound>max_upperbound): ##?These lines compare the calculated upper bound with the current max_upperbound. If the upper bound is greater, it updates max_upperbound and sets ad to the current ad index.
                max_upperbound=upper_bound
                ad=i
        ads_selected.append(ad)
        no_of_selections[ad] += 1
        reward=dataset.values[n,ad] 
        sum_of_rewards[ad]=sum_of_rewards[ad]+reward ###?sum_of_rewards: This is a list where each element corresponds to a specific ad. It keeps track of the total reward accumulated for each individual ad. For example
        total_reward=total_reward+reward         ###?This is a single variable that keeps track of the cumulative reward obtained from all ads across all rounds. It represents the overall performance of the ad selection strategy.      
    #####* Plotting the Final Graph ##############
plt.hist(ads_selected)
plt.title('Histogram of ads Selected ')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected ')
plt.show()
    