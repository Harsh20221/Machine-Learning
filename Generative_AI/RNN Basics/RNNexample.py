
##* This is a small example of RNN on a single sentence without the datasdet 

from tensorflow.keras.preprocessing.text import one_hot  # type: ignore
text=['the glass of milk',
     'the glass of juice',
     'the cup of tea',
    'I am a good boy',
     'I am a good developer',
     'understand the meaning of words',
     'your videos are good',]
voc_size=10000
###* ONE HOT ENCODING To conevrt Words into Vectors
one_hotrepr = []
for sentence in text:
    encoded = one_hot(sentence, voc_size)
    one_hotrepr.append(encoded)                                       
#print(one_hotrepr)
from tensorflow.keras.layers import Embedding #type:ignore  
from tensorflow.keras.utils import pad_sequences#type:ignore 
from tensorflow.keras.models import Sequential#type:ignore
import numpy as np
###* CREATING A EMBEDDING LAYER 
###?It converts the sparse one-hot vectors (vocabulary size of 100000) into dense vectors of size 10 (dim=10)
###?he embeddings can capture semantic relationships between words. For example, "cup" and "glass" might end up with similar embedding vectors because they are both containers
text_length=8
embedded_docs=pad_sequences(one_hotrepr,padding='pre',maxlen=text_length) ##?The Padsequences will normalize all words to be of same size , it'll either add zeroes in the start or the end so that all words are of same size  
#print(embedded_docs)
dim=10
model = Sequential()
model.add(Embedding(voc_size, dim, input_length=text_length))
# Build the model with explicit input shape
model.build((None, text_length))
model.compile('adam', 'mse')
model.summary()
model.predict(embedded_docs)
