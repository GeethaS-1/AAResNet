# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 12:41:46 2022

@author: geeth
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from datetime import datetime 
import pickle
import cv2
from PIL import Image
import keras
from keras import backend, optimizers
from keras.utils.np_utils import normalize
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
######################
from sklearn.model_selection import KFold
##########################

#image_directory ="D:/Geetha/Kvasir/augument/trainimages/"
#mask_directory ="D:/Geetha/Kvasir/augument/trainmasks/"
image_directory ="D:/Geetha/Kvasir/augument/Kvasir_trainimages/"
mask_directory ="D:/Geetha/Kvasir/augument/Kvasir_trainmasks/"
if not os.path.exists("files"):
    os.mkdir("files")
SIZE = 256
image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

images = os.listdir(image_directory)
for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
    if (image_name[-3:] == 'jpg'):
        #print(image_directory+image_name)
        image = cv2.imread(image_directory+image_name, 1)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))
masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    if (image_name[-3:] == 'jpg'):
        image = cv2.imread(mask_directory+image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image))
        #Normalize images
#image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1),3)
##D not normalize masks, just rescale to 0 to 1.
#mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.
i
#Normalize images
image_dataset = np.array(image_dataset)/255.
#D not normalize masks, just rescale to 0 to 1.
mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.
#####################
from attentionmodels import Attention_ResUNet, UNet, Attention_UNet, dice_coef, jacard_coef,precission,recall

from focal_loss import BinaryFocalLoss
from sklearn.model_selection import train_test_split
n_splits=3
kf = KFold(n_splits,shuffle=True, random_state=42)
oos_y = []
oos_pred = []
cvscores = [] 
fold = 0
for train_index, test_index in kf.split(image_dataset, mask_dataset):
#for train_index, test_index in kf.split(image_dataset, mask_dataset):
#    X_train, X_test, y_train, y_test = image_dataset[train_index], image_dataset[test_index], \
#                                     mask_dataset[train_index], mask_dataset[test_index]:
    
    fold+=1
    print(f"Fold #{fold}")
        
    X_train = image_dataset[train_index]    
    y_train = mask_dataset[train_index]
    X_test = image_dataset[test_index]
    y_test = mask_dataset[test_index]    
    IMG_HEIGHT = X_train.shape[1]
    IMG_WIDTH  = X_train.shape[2]
    IMG_CHANNELS = X_train.shape[3]
    num_labels = 1  #Binary
    input_shape = (IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
    batch_size =4
    
    Attention_ResUNet_model = Attention_ResUNet(input_shape)
    Attention_ResUNet_model.compile(optimizer=Adam(learning_rate  = 1e-2), loss=BinaryFocalLoss(gamma=2), 
              metrics=['accuracy', dice_coef, jacard_coef,precission,recall])
    callbacks = [
          ModelCheckpoint("E:/Geetha/Kvasirc/kfoldanalysis/res_attent_unet_kvasir_3fold%s.h5"%(str(fold)), save_best_only=True),
          #ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
          CSVLogger("E:/Geetha/Kvasir/kfoldanalysis/res_attent_unet_kvasir_3fold_100epochs%s.csv"%(str(fold))),
          TensorBoard(),
#         EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
    ]
    #model.fit(X_train, y_train, batch_size=8, nb_epoch=500, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint, early_stopping, adjust_learning_rate])
    Attention_ResUNet_model.fit(X_train, y_train, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test ), 
                    shuffle=True,
                    epochs=100,callbacks=callbacks)
    Attention_ResUNet_model.save('E:/Geetha/CVCClinic/kfoldanalysis/res_attent_unet_kvasir_3fold_17-6%s.h5')
    print('evaluate test data')
    score = Attention_ResUNet_model.evaluate(x = X_test, y= y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('jacard_coef:', score[2])
    print('dice_coef:', score[3])
    cvscores.append(score)
print('Mean score with std deviation over {0} folds'.format(n_splits))
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
# saving eval
with open('eval.pickle', 'wb') as handle:
   pickle.dump(cvscores, handle)