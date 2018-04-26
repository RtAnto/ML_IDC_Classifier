# -*- coding: utf-8 -*-
"""
@author: ryant
"""
from __future__ import print_function
from glob import glob
import numpy as np
from os.path import basename
import fnmatch
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

import keras
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

#Path of the files:
path = r'C:\Users\ryant\OneDrive\Documents\AlexNet\IDC_regular_ps50_idx5\*\*\*.png'
width, height = 50, 50
batch_size = 128
num_classes = 2
epochs = 12
input_shape = (50,50,3)
pictures = glob(path)

#Separate images based on classification
nonIDC =  fnmatch.filter(pictures, "*class0*")
IDC = fnmatch.filter(pictures, "*class1*")

x = []
y = [] #labels
print("About to read images:\nreading")

for i in pictures[0:10000]:
    image = cv2.imread(i)
    x.append(cv2.resize(image, (width, height),interpolation=cv2.INTER_CUBIC))
    if i in nonIDC:
        y.append(0)
    if i in IDC:
        y.append(1)

df = pd.DataFrame()
df["images"] = x
df["labels"] = y
x=np.array(x)
x=x/255.0

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)

xtrain_npArray = np.array(X_train)
xtest_npArray = np.array(X_test)
ytrain_npArray = np.array(Y_train)
ytest_npArray = np.array(Y_test)

# convert class vectors to binary class matrices
ytrain_npArray = keras.utils.to_categorical(ytrain_npArray, num_classes)
ytest_npArray = keras.utils.to_categorical(ytest_npArray, num_classes)


'''

Convolutional Layers 1-5

'''

model = Sequential()
#Layer 1, 5x5 with a stride of 4
model.add(Conv2D(96, kernel_size=7, strides = 1, padding ='same',
                 activation = 'relu', 
                 input_shape=input_shape))
# Output is now size 43x43x96
model.add(MaxPooling2D(pool_size=(3, 3), strides = 1))
model.add(BatchNormalization())
#Layer 2 # Model is now size 40x40x256
model.add(Conv2D(256, kernel_size=5,strides =1, padding='same',
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides = 1))
model.add(BatchNormalization())
model.add(Conv2D(384, kernel_size=2,strides = 1, padding='same',
                 activation = 'relu'))

#Layer 4
model.add(Conv2D(384, kernel_size=2,strides = 1, padding='same',
                 activation = 'relu'))

#Layer 5
model.add(Conv2D(256, kernel_size=2,strides = 1, padding='same',
                 activation = 'relu'))

print("Finished Layer 5")
#Final MaxPooling
model.add(MaxPooling2D(pool_size=(3, 3), strides = 2))

#Final Fully Connected Layers
model.add(Dropout(0.5))
model.add(Flatten())

model.add(Dense(4096, activation='relu'))

model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))
print("We are about to compile the Model")
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
print("We have compiled")
model.fit(xtrain_npArray, ytrain_npArray,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(xtest_npArray, ytest_npArray))

score = model.evaluate(xtest_npArray, ytest_npArray, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])