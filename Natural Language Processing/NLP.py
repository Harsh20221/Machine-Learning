import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3) ###? The delimeter is specified here as /t because it is used to read tabular seperated values , The delimeter specifies what type of value you are reading, The quoting is used to transform all the commas in the sentences to 3 because we  do not wat commas in our dataset as it may lead to problems 
###* CLEANING THE TEXT FROM THE DATASET SO THAT OUR MODEL CAN PROCESS
##/We'll be removing all non-relevant words from the dataset
import re   ###?The stemming will simplify the sentences , Whether you say " I loved this  restaurant or I love this restaurant , the meaning remains  the same so we transform that loved to love

#####!!!!VERY IMP STEP , HERE WE ARE BYPASSING SSL CERTIFICATE REQUIREMENTS , IF NOT DONE IT'LL SHOW SSL ERRORS 
import ssl
import certifi
ssl._create_default_https_context = ssl._create_unverified_context


import nltk    ###?The initial step in our cleaning process will involve removing all punctuation marks from our code
nltk.download('stopwords')
from nltk.corpus import stopwords###/This set of stopwords will contain all the words of english and based on this set we will omit other words for stemming 
from nltk.stem.porter import PorterStemmer###/ This class will be going to be used for stemming
corpus=[]##?corpus will store the cleaned reviews
for i in range(0,1000):
    reviews=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])##?Here we'll mention that we will be replacing the commas and other punctuations with space , the '[^a-zA-Z]' means that we are telling telling it to remove everything except letters from a-z & A-Z both lowercase and uppercase ,The Third Argument in the re.sub will tell where to pull the data from 
    reviews=reviews.lower() ##? This will convert all the reviews to lowercase
    reviews=reviews.split() ##? This will split the reviews
    ####/ ----- NOW WE WLL BE DOING STEMMING THAT WILL SIMPLIFY EACH OF THE WORDS INTO THEIR SIMPLER FORM , For ex--The stemming will simplify the sentences , Whether you say " I loved this  restaurant or I love this restaurant , the meaning remains  the same so we transform that loved to love
    ps=PorterStemmer()
    reviews= [ps.stem(word) for word in  reviews if not word in set(stopwords.words('english'))] ##?WE ARE APPLYING STEMMING HERE , we have enclosed this in the format of list for easier conversion , we'll convert them back to string in the next step 
    reviews=' '.join(reviews) 
    corpus.append(reviews)##? Finally appending all the clean reviews to the corpus array 
print(corpus)
 
####* Creating Bag of Words Model 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,-1].values





