
#*Importing the Libraries 
import tensorflow as tf 
#!! We have typed #type ignore because the system was giving a false error regarding the tensorflow issue , In reality is is Working Fine 
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore 
#* PREPROCESSING THE IMAGES
#*----------------------------------------------
#*preprocessing training set 
train_datagen=ImageDataGenerator(
    rescale=1./255, ##?First we are applying feature scaling to every image by dividing it's pixel value by 255
    shear_range=0.2,
    zoom_range=0.2,
horizontal_flip=True
)
##??This train_datagen.flow connects the images of our training set with the Image Data getenarator for processing
training_set=train_datagen.flow_from_directory( 
'dataset/training_set',
  target_size=(150,150),
  batch_size=32,
  class_mode='binary'  
)

##*preprocessing test set   
test_datagen=ImageDataGenerator(
    rescale=1./255,  ##?We only rescale the test set images and we do not sheer or zoom or flip it because we did that earlier to avoid overfitting and in test set this is not a problem 
)
test_set=test_datagen.flow_from_directory(
'dataset/test_set',
  target_size=(150,150), 
  batch_size=32,
  class_mode='binary'  
)    
#*Check if a trained model already exists-- We have added a additional check if the trained model already exist then use the already trained model
import os
if os.path.exists('trained_cnn_model.h5'):
    print("Loading existing trained model...")
    cnn = tf.keras.models.load_model('trained_cnn_model.h5')#?This .h5 is the model file
    print("Model loaded successfully!")
else:
    print("No existing model found. Creating and training new model...")   
##* INITIALIZE THE CNN
    cnn = tf.keras.models.Sequential()   
    #*STEP-1 CONVOLUTION 
    cnn.add(tf.keras.layers.Conv2D(filters=12,kernel_size=3,activation='relu',input_shape=[150,150,3])) ###?Here the 150,150 describes image size and 3 at last means we wanna process colured images which contain R,G,B
    ###*STEP-2 POOLING 
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2)) ###?The pool size is the size of submatrix that wthe model chooses from the Feature Map for pooling by 2 we mean 2x2 submatrix from the 5x5 matrix of the Feature map , BY strides we mean how ofen we plan on shifting the submatrix to select differnt submatrix  
    ##* Adding Second Convolution Layer 
    cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))##?The filters is the amount of feature detectors , the kernel will be 3 
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
    ##*Flattening 
    cnn.add(tf.keras.layers.Flatten())
    ##*Fully Connected layer(ANN layer)
    cnn.add(tf.keras.layers.Dense(units=200,activation='relu')) ###?We are 
    ##*OUTPUT LAYER 
    cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid')) ###?In output we use sigmoid activation function since we are doing binary classification  but if we were doing multiple classificcation we would have used Softmax Activation Function
    ##? We only need one neuron on the output because we just wanna diffrentiate between Cat or Dog 
###* Compiling the CNN 
    cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
##*Training the CNN ON Training Set and evaluating on test set
    cnn.fit(x=training_set,validation_data=test_set,epochs=30) ###?Adding training and test set , Default value for epochs is 25 but we have taken 30 for better accuracy , Epochs mean the no of times our model will run , If you increase it more then overfitting will happen , 30 is the sweet spot for this project

#* We Have addded another method to  Save the trained model
    cnn.save('trained_cnn_model.h5')
    print("Model saved as 'trained_cnn_model.h5'")

###*Making a Single Prediction
import numpy  as np 
from keras.preprocessing import image  #type:ignore
test_image=image.load_img('dataset/single_prediction/cat.4400.jpg',target_size=(150,150))##?The Target size will convert the image that we wanna identify dimensions to match with the training images dimension
test_image=image.img_to_array(test_image)##?THE predict method expects the input in form of an array so we need to convert the input image that we are testing our model on in form of an Array
test_image=np.expand_dims(test_image,axis=0)##?Adding extra dimension in our image because our  model expects a test image inside a  batch , by axis=0 we mean we want the extra dimension to  be the first one 
result = cnn.predict(test_image)
training_set.class_indices###?training_set.classindices line tells us if out model will predict 0 for cat or dog or 1 for car or dog basically we wanna know act number is for cat and what number is for dog in our prediction
if result[0][0] == 1: prediction = 'dog' ##?We are accessing result variable like this result[0][0] because we put the result variable inside a batch and added extra dimension in it so we need to access it this way 
else: prediction='cat'
#*Print the results
print(f"Class indices: {training_set.class_indices}")
print(f"Prediction probability: {result[0][0]}")
print(f"Final prediction: {prediction}")




