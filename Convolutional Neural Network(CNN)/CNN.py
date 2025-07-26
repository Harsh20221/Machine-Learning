
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
training_set=train_datagen.flow_from_dictionary( 
'dataset/training_set',
  target_size=(150,150),
  batch_size=32,
  class_mode='binary'  
)

##*preprocessing test set   
test_datagen=ImageDataGenerator(
    rescale=1./255,
)
training_set=train_datagen.flow_from_dictionary(
'dataset/training_set',
  target_size=(150,150),
  batch_size=32,
  class_mode='binary'  
)    

##* INITIALIZE THE CNN
cnn= tf.keras.models.Sequential()

#*STEP-1 CONVOLUTION 
cnn.add(tf.keras.layers.Conv2D(filters=12,kernel_size=3,activation='relu',input_shape=[150,150,3]))

###*STEP-2 POOLING 
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
##* Adding Second Convolution Layer 
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
##*Flattening 
cnn.add(tf.keras.layers.Flatten())
##*Fully Connected layer or ANN layer
cnn.add(tf.keras.layers.Dense(units=200,activation='relu')) ###?We are 
##*OUTPUT LAYER 
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))