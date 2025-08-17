
###* Initializing the Simple RNN and processing the data using RNN---Here in tghis file we are working on the IMDB dataset and we areb using RNN Learning principle to classify the reviews into  Positive or Negative 
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb #type:ignore
from tensorflow.keras.preprocessing import sequence #type:ignore
from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense #type:ignore
max_features=10000 #?Vocabulary size
(X_train,Y_train),(X_test,Y_test)=imdb.load_data(num_words=max_features)
print(f'Training Data Shape: {X_train.shape}, Training labels shape : {Y_train.shape}')##? the f after print makes this a strin g literal , Here In this step we are printing the shape of Training data Parameters that it'll be trained and parameter that  it'll predict 
print(f'Testing Data Shape: {X_test.shape}, Testing labels shape : {Y_test.shape}')##? Doing the above for test data as well 
print(X_train[0],Y_train[0])
from tensorflow.keras.preprocessing import sequence #type:ignore
max_length=200  #!! Reduced from 500 to 200 as most reviews don't need 500 words to convey sentiment
X_train=sequence.pad_sequences(X_train, maxlen=max_length, truncating='post', padding='post')
X_test=sequence.pad_sequences(X_test, maxlen=max_length, truncating='post', padding='post')
##* Train Simple RNN 
model = Sequential()
model.add(Embedding(max_features, 128, input_length=max_length))
model.add(SimpleRNN(128, activation='relu'))
model.add(Dense(1, activation='sigmoid')) ###? The dense layer is like a output layer 

# Compile the model with appropriate optimizer and loss function for binary classification
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Build the model with explicit input shape
model.build((None, max_length))

# Show model summary
model.summary()

from tensorflow.keras.callbacks import EarlyStopping #type:ignore
earlystopping=EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)

history=model.fit(
    X_train,Y_train,
    epochs=10,
    batch_size=128,  # !!Increased batch size for faster training
    validation_split=0.2,
    callbacks=[earlystopping]
)

