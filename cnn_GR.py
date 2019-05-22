 #####################################################################################
# Creator     : Gaurav Roy
# Date        : 22 May 2019
# Description : The code performs Convolutional Neural Network algorithm on the 
#               given dataset of cats and dogs. 
#####################################################################################

# Building CNN

# Importing Keras Libraries and Packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

# Initializing the CNN
classifier = Sequential()

# Step 1 = CONVOLUTION
# Feature detectors of 3x3 dimensions
# Input shape is the shape to which images need to be converted before running through CNN
# Therefore, image shape has to be of size 64px x 64px with 3 channels since RGB
classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))


# Step 2 - POOLING (MAX)
# Pool size is the size of the small subtable that is used to select values from the Feature Map 
# Pool size is 2x2 matrix and STRIDE is same as pool size by default
classifier.add(MaxPooling2D(pool_size = (2, 2)))
 
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
 
# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
 
# Step 3 - Flattening
classifier.add(Flatten())
 
# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
 
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
 
# Part 2 - Fitting the CNN to the images
# Taken from : https://keras.io/preprocessing/image/   under   .flow_from_directory(directory)
from keras.preprocessing.image import ImageDataGenerator
 
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
 
test_datagen = ImageDataGenerator(rescale = 1./255)
 
# Target size has to be same as input shape from the CNN building step
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
 
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'binary')

# steps_per_epoch = samples_per_epoch/batch_size = 8000/32 = 250
# validation_steps = nb_val_samples/batch_size = 2000/32 = 62.5 ~ 62
classifier.fit_generator(training_set,
                         steps_per_epoch = 250,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 62)