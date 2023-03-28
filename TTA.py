# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 14:16:32 2022
https://www.depends-on-the-definition.com/test-time-augmentation-keras/
@author: geeth
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")


from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.optimizers import Adam,Nadam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


from datetime import datetime 
import cv2
from PIL import Image
from keras import backend, optimizers

import os
import random
from tqdm import tqdm_notebook, tnrange
#from tqdm import tqdm.notebook.tqdm,tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
im_width = 128
im_height = 128
border = 5
path_train = 'D:/Geetha/TEST/images/Train_test/'
#path_test = '../input/test/'

#path_train = 'D:/Geetha/TEST/images/Train_test1/'
#Load images

# Get and resize train images and masks
def get_data(path, train=True):
    ids = next(os.walk(path + "images"))[2]
    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    if train:
        y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    print('Getting and resizing images ... ')
    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        # Load images
        img = load_img(path + '/images/' + id_, grayscale=True)
        x_img = img_to_array(img)
        x_img = resize(x_img, (128, 128, 1), mode='constant', preserve_range=True)

        # Load masks
        if train:
            mask = img_to_array(load_img(path + '/masks/' + id_, grayscale=True))
            mask = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)

        # Save images
        X[n, ..., 0] = x_img.squeeze() / 255
        if train:
            y[n] = mask / 255
    print('Done!')
    if train:
        return X, y
    else:
        return X
    
X, y = get_data(path_train, train=True)
#image_directory = "D:/TEST/images/train/"
#mask_directory = "D:/TEST/images/trainmask/"
##"D:/Geetha/EETISLarib/augument/trainmasks/"
#if not os.path.exists("files"):
#    os.mkdir("files")
#SIZE = 256
#image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
#mask_dataset = [] 
############################
#
#images = os.listdir(image_directory)
#for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
#    if (image_name[-3:] == 'jpg'):
#        #print(image_directory+image_name)
#        image = cv2.imread(image_directory+image_name, 1)
#        image = Image.fromarray(image)
#        image = image.resize((SIZE, SIZE))
#        image_dataset.append(np.array(image))
#
##Iterate through all images in Uninfected folder, resize to 64 x 64
##Then save into the same numpy array 'dataset' but with label 1
#
#masks = os.listdir(mask_directory)
#for i, image_name in enumerate(masks):
#    if (image_name[-3:] == 'jpg'):
#        image = cv2.imread(mask_directory+image_name, 0)
#        image = Image.fromarray(image)
#        image = image.resize((SIZE, SIZE))
#        mask_dataset.append(np.array(image))
#        image_dataset = np.array(image_dataset)/255.
##D not normalize masks, just rescale to 0 to 1.
#mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=2018)

#import random
#import numpy as np
#image_number = random.randint(0, len(X_train))
#plt.figure(figsize=(12, 6))
#plt.subplot(121)
#plt.imshow(np.reshape(X_train[image_number], (256, 256, 3)), cmap='gray')
#plt.subplot(122)
#plt.imshow(np.reshape(y_train[image_number], (256, 256 )), cmap='gray')
#plt.show()
# Check if training data looks all right
ix = random.randint(0, len(X_train))
has_mask = y_train[ix].max() > 0

fig, ax = plt.subplots(1, 2, figsize=(20, 10))

ax[0].imshow(X_train[ix, ..., 0], cmap='seismic', interpolation='bilinear')
if has_mask:
    ax[0].contour(y_train[ix].squeeze(), colors='k', levels=[0.5])
ax[0].set_title('Seismic')

ax[1].imshow(y_train[ix].squeeze(), interpolation='bilinear', cmap='gray')
ax[1].set_title('Salt');
#Parameters for model

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]
num_labels = 1  #Binary
input_shape = (IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
#batch_size = 8

from focal_loss import BinaryFocalLoss
#Try various models: Unet, Attention_UNet, and Attention_ResUnet
#Rename original python file from 224_225_226_models.py to models.py
from attentionmodels import Attention_ResUNet, UNet, Attention_UNet, dice_coef, dice_coef_loss, jacard_coef,jacard_coef_loss,focal_tversky_loss,sensitivity,specificity,confusion,true_positive,true_negative
att_res_unet_model = Attention_ResUNet(input_shape)

att_res_unet_model.compile(optimizer=Adam(learning_rate  = 1e-2), loss=BinaryFocalLoss(gamma=2), 
              metrics=['accuracy', jacard_coef , dice_coef])



data_gen_args = dict(horizontal_flip=True,
                     vertical_flip=True)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 2018
bs = 8

image_generator = image_datagen.flow(X_train, seed=seed, batch_size=bs, shuffle=True)
mask_generator = mask_datagen.flow(y_train, seed=seed, batch_size=bs, shuffle=True)



# Just zip the two generators to get a generator that provides augmented images and masks at the same time
train_generator = zip(image_generator, mask_generator)

callbacks = [
#    EarlyStopping(patience=10, verbose=1),
#    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.000001, verbose=1),
    ModelCheckpoint('E:/Geetha/Revision_1/TTAanalysis/TTAtest.h5', verbose=1, save_best_only=True, save_weights_only=True),
    CSVLogger("E:/Geetha/Revision_1/TTAanalysis/ttatest%s.csv"),
#    EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
]

results = att_res_unet_model.fit(train_generator, steps_per_epoch=(len(X_train) // bs), epochs=90, callbacks=callbacks,
                              validation_data=(X_valid, y_valid))
plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();
# Load best model
att_res_unet_model.load_weights('E:/Geetha/Revision_1/TTAanalysis/TTAtest.h5')
# Evaluate on validation set (this must be equals to the best log_loss)
att_res_unet_model.evaluate(X_valid, y_valid, verbose=1)
