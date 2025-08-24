import nltk 
import pandas as pd 
import numpy as np
import os

nltk.download('gutenberg')
from nltk.corpus import gutenberg

#* Load the Dataset
data=gutenberg.raw('shakespeare-hamlet.txt')
#*Saving the Dataset
with open('hamlet.txt','w') as file:
    file.write(data)
  
#* THE ACTUAL TRAINING OF THE MODEL STARTS FROM BELOW     
    
#* Preprocessing 
from tensorflow.keras.preprocessing.text import Tokenizer #type:ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences#type:ignore
from sklearn.model_selection import train_test_split #type:ignore
#* load the Dataset
with open ('hamlet.txt','r') as file:
    text=file.read().lower()
    
#*Tokenize the Text
tokenizer=Tokenizer()
tokenizer.fit_on_texts([text])
total_words=len(tokenizer.word_index)+1  ##?Tokenizing total words
###print(total_words)
###print(tokenizer.word_index)

#*Create Input Sequences
input_sequences=[]
for line in text.split('\n'):
    token_list=tokenizer.texts_to_sequences([line])[0] #?It is converting every word into sentence line by line
    for i in range (1,len(token_list)):
        n_gram_sequence=token_list[:i+1]
        input_sequences.append(n_gram_sequence)
#* Applying Pad Sequences to make sure we get all the sequences of equal length
max_sequence_len=max(len(x) for x in input_sequences) ##? This line will find which sentence has the maximum length 
##print(max_sequence_len)        
input_sequences=np.array(pad_sequences(input_sequences,maxlen=max_sequence_len,padding='pre')) ##? Here  we are applying the padding where we are making all sentence equal to the size of maximul length sentence by adding zeroes before the shorter sentences 
    
#*Create Predictors and Labels and Divide into training and test sets 
import tensorflow as tf 
x,y=input_sequences[:,:-1],input_sequences[:,-1] ##? Here we are seperating x and y where x refers to our parameters and y refers to what we want to predict , in x we take ever column except the last while in y we only take the last column
y=tf.keras.utils.to_categorical(y,num_classes=total_words) ##? Converting y(output) to categories where total categories is no of all istinct words in the Hamlet 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)##?Splitting traing and Test Set 



###*THIS WILL CHECK FOR AVAILABLE MODELS AND IF found it'll use that 
model_path = 'next_word_lstm.h5'

if os.path.exists(model_path):
    print(f"Loading existing model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
#* Training our Model
  ##*Defining the Model
else:  
    from tensorflow.keras.models import Sequential#type:ignore
    from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout#type:ignore  ##? Embedding represent words as meaningful numerical vectors. Computers don't understand words like "cat" or "king"; they only understand numbers. An embedding layer converts categorical data, like words, into dense, low-dimensional vectors that capture semantic relationships
    model=Sequential()
    model.add(Embedding(total_words,100,input_length=max_sequence_len-1)) ##?here 100 is the dimension , we have subtracted -1 from max_sequence length because we are starting the measurment of length from zero and to get the actual length we are subtracting one 
    model.add(LSTM(150,return_sequences=True)) ##?Here 150 is the no of neurons 
    model.add(Dropout(0.2))#?The dropout layer disables some  hidden neurons when training to prevent overfitting
    model.add(LSTM(100))
    model.add(Dense(total_words,activation='softmax')) ##?This as the model's decision-making layer. Before this point, the network (likely LSTMs or other recurrent layers) has processed the input sequence and condensed its understanding into a set of internal numerical representations. The Dense layer takes this final representation and maps it to a score for every single word that could possibly be the next word.The total_words argument is crucial. It tells the Dense layer how many output neurons to create—one for each word in your vocabulary.
    model.build(input_shape=(None, max_sequence_len)) ###!!!!VERY IMPORTANT TO EXPLICITLY BUILD THE MODEL , THIS IS A ADDITIONAL STEP NOT IN THE TUTORIALS , IF IT'LLL BE SKIPPED THEN THERE WILL EVERYTHING BE ZERO AND UNBUILT IN MODEL SUMMARY, The max_sequence_length is the maximum length of any sentence 
  #* Compile the Model
    model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy']) 
    model.summary()

#* Train the Model 
    history=model.fit(x_train,y_train,epochs=100,validation_data=(x_test,y_test),verbose=1) ##Run for 100 epochs atleast for good accuracy 

#*Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None
##*Testing the model by predicting the next word based on our own sentence 
### input_text=" Barn. How now Horatio? You tremble & look "
###print(f"Input text:{input_text}")
##max_sequence_len=model.input_shape[1]+1
##next_word=predict_next_word(model,tokenizer,input_text,max_sequence_len)
##print(f"Next Word PRediction:{next_word}") 


##*Saving the Model
model.save("next_word_lstm.h5")
##* Save the tokenizer
import pickle
with open('tokenizer.pickle','wb') as handle:
    pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)
##* Creating a Streamlit App
import streamlit as st  ###!!To run this app on WSL--streamlit run LSTMRNN.py --server.address 0.0.0.0
st.title("Next Word predictor with LSTM")
input_text=st.text_input("Enter the Sequence of words","To be or not to")
if st.button("Predict Next Word"):
    max_sequence_len=model.input_shape[1]+1#?Retrieve the max sequence length 
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len) ##!!Make sure to not firget to add the predict ststement here again
    st.write(f'Next Word: {next_word}')
    





