# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import keras
import matplotlib.pyplot as plt
from glob import glob
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, ZeroPadding2D, Conv2D, MaxPooling2D,Input,SeparableConv2D
from keras.preprocessing.image import ImageDataGenerator #Data augmentation and preprocessing
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
import cv2
from PIL import Image
from pathlib import Path
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,recall_score,confusion_matrix



# Exploring the directories in our dataset
print(os.listdir("../input/chest_xray/chest_xray"))

path_train = "../input/chest_xray/chest_xray/train"
path_val = "../input/chest_xray/chest_xray/val"
path_test = "../input/chest_xray/chest_xray/test"



# Example plots of images in NORMAL and PNEOMONIA folder
plt.figure(1, figsize = (15 , 7))
plt.subplot(1 , 2 , 1)
img = glob(path_train+"/Conid-19/*.jpeg") #Getting an image in the PNEUMONIA folder
img = np.asarray(plt.imread(img[0]))
plt.title('Covid-19 X-RAY')
plt.imshow(img)

plt.subplot(1 , 2 , 2)
img = glob(path_train+"/NORMAL/*.jpeg") #Getting an image in the NORMAL folder
img = np.asarray(plt.imread(img[0]))
plt.title('NORMAL CHEST X-RAY')
plt.imshow(img)

plt.show()



# AUGMENTATION ON TRAINING, VALIDATION, TEST DATA

train_gen = ImageDataGenerator(rescale = 1./255,
                             shear_range = 0.2,
                             zoom_range = 0.2,
                             horizontal_flip=True)

val_gen = ImageDataGenerator(rescale=1./255)

train_batch = train_gen.flow_from_directory(path_train,
                                            target_size = (64, 64),
                                            classes = ["NORMAL", "Covid-19"],
                                            class_mode = "categorical")
val_batch = val_gen.flow_from_directory(path_val,
                                        target_size = (64, 64),
                                        classes = ["NORMAL", "Covid-19"],
                                        class_mode = "categorical")
test_batch = val_gen.flow_from_directory(path_test,
                                         target_size = (64, 64),
                                         classes = ["NORMAL", "Covid-19"],
                                         class_mode = "categorical")

print(train_batch.image_shape)



# Creating the CNN model


model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation="relu",input_shape=(64,64,3)))

model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),activation="relu"))

model.add(Flatten())
model.add(Dense(128,activation="relu"))

model.add(Dense(1,activation="sigmoid"))


#Function to create the accuracy and loss plots

def create_plots(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


model = build_model()
model.summary()



#  Creating 4 callbacks to monitor the accuracy and loss and get the best model

batch_size = 32
epochs = 20
early_stop = EarlyStopping(patience=25,
                           verbose = 2,
                           monitor='val_loss',
                           mode='auto')

checkpoint = ModelCheckpoint(
    filepath='best_model',
    save_best_only=True,
    save_weights_only=True,
    monitor='val_loss',
    mode='auto',
    verbose = 1)

reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.8,
    patience=5,
    verbose=1,
    mode='auto',
    min_delta=0.0001,
    cooldown=1,
    min_lr=0.0001
)


# Compiling the model.
# Binary crossentropy is used because its a binary image classification problem.
# You can vary the number of epochs,etc to get more extensive plots but it will make the program more computationally expensive.

model.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer=Adam(lr=0.001))

history = model.fit_generator(epochs=epochs,
                              callbacks=[early_stop,checkpoint,reduce],
                              shuffle=True,
                              validation_data=val_batch,
                              generator=train_batch,
                              steps_per_epoch=100,
                              validation_steps=20,
                              verbose=2)



create_plots(history)


# Get the original labels from the test folder
# The try,catch statements are used to avoid the errors during resizing.
# The labels from NORMAL and PNEUMONIA are one hot encoded using to_categorical.
#

original_test_label = []
images = []
test_pneumonia = Path("../input/chest_xray/chest_xray/test/Covid-19")
pneumonia = test_pneumonia.glob('*.jpeg')
for i in pneumonia:
    img = cv2.imread(str(i))
    #     print("pneumonia",img)
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        img = cv2.resize(img, (64, 64))
    except Exception as e:
        print(str(e))
    images.append(img)
    label = to_categorical(1, num_classes=2)
    original_test_label.append(label)

test_normal = Path("../input/chest_xray/chest_xray/test/NORMAL")
normal = test_normal.glob('*.jpeg')
for i in normal:
    img = cv2.imread(str(i))
    #     print("normal",img)
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        img = cv2.resize(img, (64, 64))
    except Exception as e:
        print(str(e))
    images.append(img)
    label = to_categorical(0, num_classes=2)
    original_test_label.append(label)

images = np.array(images)
original_test_label = np.array(original_test_label)
print(original_test_label.shape)


# The prediction of labels using the test data

p = model.predict(images, batch_size=32)
preds = np.argmax(p, axis=-1)
print(preds.shape)

# Evaluation of the model to calculate loss, score.

test_loss, test_score = model.evaluate(images, original_test_label, batch_size=16)
print("Loss on test set: ", test_loss)
print("Accuracy on test set: ", test_score)

# Calculating the confusion matrix , recall score and auc score

orig_test_labels = np.argmax(original_test_label, axis=-1)
print(orig_test_labels.shape)
confusion_matrix(orig_test_labels,preds)
recall_score(orig_test_labels,preds)
roc_auc_score(orig_test_labels,preds)

