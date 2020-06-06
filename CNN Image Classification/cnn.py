#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 12:53:14 2020

@author: amitkumar
"""

#Import the libraries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

#Build the sequential model
classifier = Sequential()

#Build the Convolutional layer
classifier.add(Conv2D(32,(3,3), input_shape=(64,64,3), activation='relu'))

#Build the Pooling layer
classifier.add(MaxPool2D(pool_size=(2,2)))

#Build the Flatten layer
classifier.add(Flatten())

#Build the Dense layer
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

#Compile the Classifier
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fitting the data to the CNN model
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(directory='/Users/amitkumar/Documents/Projects/Deployed Model/CNN Image Classification',
                                                 target_size=(64,64),
                                                 batch_size=32,
                                                 class_mode='binary')
testing_set = test_datagen.flow_from_directory(directory='/Users/amitkumar/Documents/Projects/Deployed Model/CNN Image Classification',
                                               target_size=(64,64),
                                               batch_size=32,
                                               class_mode='binary')
classifier.fit_generator(generator=training_set,
                         steps_per_epoch=8000,
                         epochs=5,
                         validation_data=testing_set,
                         validation_steps=2000)

classifier.save('model.h5')








