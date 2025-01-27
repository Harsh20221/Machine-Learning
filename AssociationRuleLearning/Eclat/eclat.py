import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
###*Importing the dataset
dataset= pd.read_csv('Market_Basket_Optimisation.csv',header=None)##?header is null because we also want to include the first line of the dataset that is why we have written header is eqiual to null
###* Inserting the elements of the dataset into a transactions array 
transactions=[]
for i in range(1,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(1,20)])
###*Preparing the Model for predicdtions  & Doing Predictions 
from apyori import apriori
rules=apriori(transactions=transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2,max_length=2)  #?transactions will be the list of all the transactions that we created for the model to process
#?We have Calculated our minimum support using the formula that we have 3 transactions everyday and the data we have is for 7 days so we multiply 3 with 7 and divide by total no of transactions that is 7501 , hence our formula will be 3*7/7501
#?In minimum confidence we specify the minimum  percentage  that a  rule   needs to be correct to get included in the calculation 
#?Lift is the Gain in the prediction , means ex---compared to suggesting someone random , how much gain in the accuracy of predictions we got, It Minimum should be 3 
#?Since we are finding out the best deals for buy one product and get another product for free so we will be taking min lift as 2 and max lift  as 2 
###*Organising the Messy  list into a well organinised Table to display ,
######/NOTE::::THE ECLAT ALGORITHM CODE WILL BE SAME AS THE APRIORI ALGORITHM CODE , THE ONLY DIFFERENCE WILL BE IN THE PARAMETERS THAT WE WILL BE PASSING TO THE APRIORI FUNCTION , WE WILL NOT BE DISPLAYING CONFIDENCE AND LIFT IN THE TABLE WE ARE MAKING BELOW , WE WILL BE DISPLAYING ONLY THE SUPPORT AND THE PRODUCTS THAT THE CUSTOMER HAS BOUGHT

results=list(rules) ##?Copying the rules to results 
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results] ##?THIS [2][0[0]... IS USED TO ACCESS THE CURRECT PARAMETERS FROM THE RESULTS LIST 
    rhs         = [tuple(result[2][0][1])[0] for result in results]##?The way we are accessing the parameters is because the parameters are in the form of a tuple and we are accessing the first element of the tuple , the first element of the tuple will be the product that the customer has bought , the second element will be the product that the customer has also bought after buying the first product, the third element will be the support , the fourth element will be the confidence and the fifth element will be the lift
    supports    = [result[1] for result in results]  ##?The support is the number of transactions that have the product in the left hand side and the right hand side of the rule divided by the total number of transactions\

    return list(zip(lhs, rhs, supports, ))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product-1', 'Product-2', 'Support']) ####/ UMLIKE APRIORI THERE WILL NOT BE ANY LEFT HAND SIDE OR RIGHT HAND SIDE IN THE ECLAT ALGORITHM , WE WILL BE DISPLAYING THE PRODUCTS THAT THE CUSTOMER HAS BOUGHT IN THE LEFT HAND SIDE AND THE RIGHT HAND SIDE OF THE RULE
###*Sorting all the rules in the table by lift parameter 
###/NOTE: UNlike apriori we will be sorting based on the support parameter in eclat algorithm not lift
resultsinDataFrame.nlargest(n=10,columns='Support')##?here n is the number of rules that we want to display , we are displaying 10 rules here
print(resultsinDataFrame)#####?THIS WILL DISPLAY THE RESULT --- THE LEFT HAND SIDE WILL HAVE THE PRODUCT THAT THE CUSTOMER FIRST BOUGHT AND RIGHT HAND SIDE WILL HAVE THE PRODUCT THAT CUSTOMER HAS ALSO BOUGHT WHEN HE/SHE HAS BOUGHT THE FIRST PRODUCT , THE CONFIDENCES WILL HAVE THE PERCENTAGE OF CUSTOMER FOR EVERY RULE IN point 



    