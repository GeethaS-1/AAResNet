pp ,# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 13:47:59 2022

@author: geeth
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from datetime import datetime 
import cv2
from PIL import Image
from keras import backend, optimizers
from keras.utils.np_utils import normalize
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard

#******************************************Augmentation code

#import Augmentor
#from Augmentor import Pipeline
# #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.
############
#p = Augmentor.Pipeline('D:/CVCClinic/augment_test/cvc_augment/trainimages/')
#
## Specify a directory containing the ground truth or mask images:
#p.ground_truth('D:/CVCClinic/augment_test/cvc_augment/trainmasks/')
#
## Add operations to the pipeline as normal:
#p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
#p.flip_left_right(probability=0.4)
#p.flip_top_bottom(probability=0.4)
#p.random_distortion(probability=0.5, grid_width=6, grid_height=6, magnitude=3)
## Sample as normal:
#p.sample(1200)
########################
image_directory = "D:/Geetha/CVCClinic/augment/trainimages/"
mask_directory = "D:/Geetha/CVCClinic/augment/trainmasks/"

if not os.path.exists("files"):
    os.mkdir("files")
SIZE = 256
image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
mask_dataset = [] 
###########################

images = os.listdir(image_directory)
for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
    if (image_name[-3:] == 'png'):
        #print(image_directory+image_name)
        image = cv2.imread(image_directory+image_name, 1)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))

#Iterate through all images in Uninfected folder, resize to 64 x 64
#Then save into the same numpy array 'dataset' but with label 1

masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    if (image_name[-3:] == 'png'):
        image = cv2.imread(mask_directory+image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image))

        #Normalize images
#image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1),3)
##D not normalize masks, just rescale to 0 to 1.
#mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.

#Normalize images
image_dataset = np.array(image_dataset)/255.
#D not normalize masks, just rescale to 0 to 1.
mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.30, random_state = 0)

#Sanity check, view few mages
import random
import numpy as np
image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (256, 256, 3)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (256, 256 )), cmap='gray')
plt.show()

#Parameters for model

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]
num_labels = 1  #Binary
input_shape = (IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
batch_size = 8

#FOCAL LOSS AND DICE METRIC
#Focal loss helps focus more on tough to segment classes.

from focal_loss import BinaryFocalLoss
#Try various models: Unet, Attention_UNet, and Attention_ResUnet
#Rename original python file from 224_225_226_models.py to models.py
from attentionmodels import Attention_ResUNet, UNet, Attention_UNet, dice_coef, dice_coef_loss, jacard_coef,jacard_coef_loss,focal_tversky_loss,sensitivity,specificity,confusion,true_positive,true_negative
att_res_unet_model = Attention_ResUNet(input_shape)

att_res_unet_model.compile(optimizer=Adam(learning_rate  = 1e-2), loss=BinaryFocalLoss(gamma=2), 
              metrics=['accuracy', jacard_coef , dice_coef, dice_coef_loss,jacard_coef_loss,focal_tversky_loss,sensitivity,specificity,confusion,true_positive,true_negative])



# att_res_unet_model.compile(optimizer=Adam(lr = 1e-3), loss='binary_crossentropy', 
#               metrics=['accuracy', jacard_coef])

print(att_res_unet_model.summary())
callbacks = [
          ModelCheckpoint("E:/Geetha/Revision_1/CVC/analysis70_30/resatten_unet_model_20-6_augCVC_earlystop.h5", save_best_only=True,verbose=1),
#          ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
          CSVLogger("E:/Geetha/Revision_1/CVC/analysis70_30/resatten_unet_model_20-6_augCVC_earlystop.csv"),
          TensorBoard(),
         EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
]

start3 = datetime.now() 
att_res_unet_history_cvc = att_res_unet_model.fit(X_train, y_train, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test ), 
                    shuffle=False,
                    epochs=200,callbacks=callbacks)
stop3 = datetime.now()

#Execution time of the model 
execution_time_AttResUnet = stop3-start3
print("Attention ResUnet execution time is: ", execution_time_AttResUnet)

