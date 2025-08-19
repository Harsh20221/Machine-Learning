
###* Initializing the Simple RNN and processing the data using RNN---Here in tghis file we are working on the IMDB dataset and we areb using RNN Learning principle to classify the reviews into  Positive or Negative 
import os 
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
max_length=500  #!! Reduced from 500 to 200 as most reviews don't need 500 words to convey sentiment
X_train=sequence.pad_sequences(X_train, maxlen=max_length, truncating='post', padding='post')
X_test=sequence.pad_sequences(X_test, maxlen=max_length, truncating='post', padding='post')

###*THIS WILL CHECK FOR AVAILABLE MODELS AND IF found it'll 
model_path = 'simple_Rnnnew_imdb.h5'

if os.path.exists(model_path):
    print(f"Loading existing model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

else:
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
    earlystopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, Y_train,
        epochs=10,
        batch_size=128,  # !!Increased batch size for faster training
        validation_split=0.2,
        callbacks=[earlystopping]
    )
model.save('simple_Rnnnew_imdb.h5')

##* Here We are doing the Predictions
##?Even though we did this preprocessing for the training data, we must repeat it for any new data you want to predict on. The model only understands numbers, not raw text
# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Step 2: Helper Functions
#? Function to decode reviews,This is here for  visualization purposes only and not functional in this code , Since the Imdb dataset stores the reviews in form of numbers so we need decoding if we wanna see those reviews as text
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

#?We are doing preprocessing again because --The model was trained on numerical data (word indices), not raw textEach review needs to be the exact same length as what the model expectsWe need to use the same word-to-index mapping that was used during training
# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)

    prediction=model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    return sentiment, prediction[0][0]

# Step 4: User Input and Prediction
# Example review for prediction
example_review = "This movie was bad"

sentiment,score=predict_sentiment(example_review)

print(f'Review: {example_review}')
print(f'Sentiment: {sentiment}')
print(f'Prediction Score: {score}')