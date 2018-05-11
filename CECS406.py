
# coding: utf-8

# In[7]:


from glob import glob
from PIL import Image
import numpy as np
from os.path import basename
import fnmatch
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import misc
import matplotlib.pyplot as plt
import imageio
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

#Path of the files:
path = r'C:\Users\dfung\OneDrive\Desktop\breast-histopathology-images\IDC_regular_ps50_idx5\*\*\*.png'
width, height = 50, 50
batch_size = 300
num_classes = 2
epochs = 50
input_shape = (50,50,3)
pictures = glob(path)

#Separate images based on classification
nonIDC =  fnmatch.filter(pictures, "*class0*")
IDC = fnmatch.filter(pictures, "*class1*")

x = []
y = [] #labels
for i in pictures[0:100000]:
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

model = Sequential()
#50x50
model.add(Conv2D(8, kernel_size=(3, 3), padding = 'same',
                 activation='relu',
                 input_shape=input_shape))

#48x48
model.add(Conv2D(8, (3, 3), activation='relu'))

#48x48
model.add(Conv2D(16, (3, 3),padding = 'same', activation='relu'))
model.add(Conv2D(16, (3, 3),padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#24x24
model.add(Conv2D(32, (3, 3),padding = 'same', activation='relu'))
model.add(Conv2D(32, (3, 3),padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#12x12
model.add(Conv2D(32, (3, 3),padding = 'same', activation='relu'))
model.add(Conv2D(32, (3, 3),padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#6x6
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(xtrain_npArray, ytrain_npArray,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(xtest_npArray, ytest_npArray))
score = model.evaluate(xtest_npArray, ytest_npArray, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

def plot_loss_accuracy(history):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history.history["loss"],'r-x', label="Train Loss")
    ax.plot(history.history["val_loss"],'b-x', label="Validation Loss")
    ax.legend()
    ax.set_title('cross_entropy loss')
    ax.grid(True)


    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history["acc"],'r-x', label="Train Accuracy")
    ax.plot(history.history["val_acc"],'b-x', label="Validation Accuracy")
    ax.legend()
    ax.set_title('accuracy')
    ax.grid(True)
    

plot_loss_accuracy(history)


# In[ ]:


y_pred = model.predict(xtest_npArray)
map_characters = { 0:'IDC(-)', 1: 'IDC(+)'}
print('/n', sklearn.metrics.classification_report(np.where(ytest_npArray > 0)[1], np.argmax(y_pred, axis = 1), target_names=list(map_characters.values())), sep='')


# In[7]:





# In[8]:




