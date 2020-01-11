
# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras.layers.core import Activation

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
image_size = 84

inputShape = (image_size, image_size, 3)
chanDim = -1

##### CONV 1 #####
classifier.add(Convolution2D(32, 3, 3, input_shape = (image_size, image_size, 3), activation = 'relu'))
classifier.add(BatchNormalization(axis=chanDim))
classifier.add(Dropout(0.25))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
# classifier.add(Dense(output_dim = 1024, activation = 'relu'))
classifier.add(Dense(1024))
classifier.add(Activation("relu"))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.5))

# classifier.add(Dense(output_dim = 5, activation = 'softmax'))

classifier.add(Dense(5))
classifier.add(Activation('softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('data_classifier/train',
                                                 target_size = (image_size, image_size),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('data_classifier/test',
                                            target_size = (image_size, image_size),
                                            batch_size = 32,
                                            class_mode = 'categorical')

classifier.fit_generator(training_set,
                         samples_per_epoch = 128,
                         nb_epoch = 25,
                         validation_data = test_set,
                         )