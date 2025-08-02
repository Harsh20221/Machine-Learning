import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt_tab') 

corpus="i am harsh , I am ** Years old , I love Nuclear Submarines , My favourite is akula class submarine "
print(sent_tokenize(corpus))###? The Tokenize helps to seperate long sentences bundled together into seperate sentences 
print(word_tokenize(corpus)) #?This will seperate all the words from a sentence
from nltk.stem import RegexpStemmer
reg_stemmer=RegexpStemmer("ing$|s$|e$|able$",min=4)##?min=4 specifies the minimum length of the word after the stemming operation has been applied
nltk.download('wordnet')##? Required to lammitize 
from nltk.stem import WordNetLemmatizer ##?The aim of Lemmitizer is to get the exact main word which is meaningful like, It is like stemming but produces a Valid word which means the same thing as the root word
lemmatizer=WordNetLemmatizer()
result=lemmatizer.lemmatize("I named my cat shirley because she used to shake while walking")
print(result)
