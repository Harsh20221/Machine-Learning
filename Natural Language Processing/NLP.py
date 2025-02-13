import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3) ###? The delimeter is specified here as /t because it is used to read tabular seperated values , The delimeter specifies what type of value you are reading, The quoting is used to transform all the commas in the sentences to 3 because we  do not wat commas in our dataset as it may lead to problems 
###* CLEANING THE TEXT FROM THE DATASET SO THAT OUR MODEL CAN PROCESS
##/We'll be removing all non-relevant words from the dataset
import re   ###?The stemming will simplify the sentences , Whether you say " I loved this  restaurant or I love this restaurant , the meaning remains  the same so we transform that loved to love
import nltk    ###?The initial step in our cleaning process will involve removing all punctuation marks from our code
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]##?corpus will store the cleaned reviews
for i in range(0,1000):
    reviews=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])##?Here we'll mention that we will be replacing the commas and other punctuations with space , the '[^a-zA-Z]' means that we are telling telling it to remove everything except letters from a-z & A-Z both lowercase and uppercase ,The Third Argument in the re.sub will tell where to pull the data from 
    reviews=reviews.lower ##? This will convert all the reviews to lowercase
    reviews=reviews.split()##? This will split the reviews






