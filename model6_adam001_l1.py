# -*- coding: utf-8 -*-
"""
Created on Wed May  4 20:15:10 2022
low dose simulation
model 6: 8 residual blocks, 2 skip connections
Adam(0.01),epoch=1000,batch=2
@author: xialumi
"""


import os
import random
import math
import shutil
import cv2
import csv
import numpy as np
import pandas as pd
import pickle

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

import skimage
from skimage import io, transform
from skimage import data, img_as_float
from skimage import exposure, morphology, transform
from skimage import filters, measure, util

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

import pydicom as dicom
from pydicom.pixel_data_handlers.util import apply_modality_lut

from PIL import Image
#import plotly
#import vtk
#import patoolib




import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Layer, Dense, Flatten, Dropout, Reshape, Input, concatenate, add
from tensorflow.keras.layers import LeakyReLU, Conv2D, MaxPool2D, GlobalAvgPool2D, BatchNormalization, Conv2DTranspose
from tensorflow.keras.layers.experimental.preprocessing import Resizing

from tensorflow.keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError 
from tensorflow.keras.optimizers import SGD, Adam, Adadelta, RMSprop

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger





def crop(img,size):
    w=img.shape[0]
    h=img.shape[1]
    cw=ch=size
    cropped_img = img[h//2 - ch//2:h//2 + ch//2, w//2 - cw//2:w//2 + cw//2]
    plt.imshow(cropped_img)
    plt.show()
    
    return cropped_img


def window_image(image, ds, window_center, window_width):

    hu = apply_modality_lut(image, ds)
    
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2

    window_image = hu.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max

    return window_image




def set_window(img,ds,window):
    
    window_option={
        "lungs": [-600,1500],
        "midastinum": [50,400],
        "abdomen": [50,250],
        "liver": [30,150],
        "bone": [400,1800]
    }
    
    return window_image(img, ds, window_option[window][0], window_option[window][1])


def pair(df1,df2):
    
    pair1=[]
    pair2=[]
    for i in range(len(df1)):
        sloc1=list(df1['slice location'])[i]
        nm1=list(df1['img'])[i]
        for j in range(len(df2)):
            sloc2=list(df2['slice location'])[j]
            nm2=list(df2['img'])[j]
            if abs(sloc1-sloc2) <= 0.5:
                pair1.append(nm1)
                pair2.append(nm2)
            
    return pair1,pair2


def readdicom(nm,dspath,window):
    
    ds=dicom.dcmread(os.path.join(dspath,nm))
    #print(ds.SliceLocation)
    img=set_window(ds.pixel_array,ds,window)
    
    return img


def readimg(df1,df2,dspath,window,sample_ratio):
    
    pair1,pair2 = pair(df1,df2)
    idxw = [i for i in range(len(pair1))]
    idxc = random.sample(idxw,round(sample_ratio*len(pair1)))
    imgnm1 = [pair1[i] for i in idxc]
    imgnm2 = [pair2[i] for i in idxc]
    
    imgs1=[]
    imgs2=[]
    for nm in imgnm1:
        img=readdicom(nm,dspath,window)
        imgs1.append(img)
        
    for nm in imgnm2:
        img=readdicom(nm,dspath,window)
        imgs2.append(img)
        
    return np.array(imgs1),np.array(imgs2)



"""
resnet
"""
#According to GitHub Repository: https://github.com/calmisential/TensorFlow2.0_ResNet
class BasicBlock(Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same")
        self.bn1 = BatchNormalization()
        self.lrelu1 = LeakyReLU(alpha=0.2)
        self.conv2 = Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same")
        self.bn2 = BatchNormalization()
        self.lrelu2 = LeakyReLU(alpha=0.2)
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(Conv2D(filters=filter_num,
                                                       kernel_size=(1, 1),
                                                       strides=stride))
            self.downsample.add(BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.lrelu1(x)
        #x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.lrelu1(x)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output
    
    
class BottleNeck(Layer):
    def __init__(self, filter_num, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = Conv2D(filters=filter_num,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = Conv2D(filters=filter_num * 4,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same')
        self.bn3 = BatchNormalization()

        self.downsample = Sequential()
        self.downsample.add(Conv2D(filters=filter_num * 4,
                                                   kernel_size=(1, 1),
                                                   strides=stride))
        self.downsample.add(BatchNormalization())

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output



def make_basic_block_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, stride=stride))

    for i in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride=1))

    return res_block


def make_bottleneck_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BottleNeck(filter_num, stride=stride))

    for i in range(1, blocks):
        res_block.add(BottleNeck(filter_num, stride=1))

    return res_block



"""
3 * conv(conv, bn, leaky relu)
n * residual block
3 * deconv(2*(resize, conv, bn, lrelu), 1*(conv,bn,lrelu))

2 skip connections

"""
def generator(res_num):

    #encoder
    inp = Input((512,512,1))
    
    conv1 = Conv2D(64,3,padding = 'same', activation = 'relu')(inp)
    bn1 = BatchNormalization()(conv1)
    lr1 = LeakyReLU(alpha=0.2)(bn1)
    
    conv2 = Conv2D(128,3,padding = 'same', activation = 'relu')(lr1)
    bn2 = BatchNormalization()(conv2)
    lr2 = LeakyReLU(alpha=0.2)(bn2)
    
    conv3 = Conv2D(256,3,padding = 'same', activation = 'relu')(lr2)
    bn3 = BatchNormalization()(conv3)
    lr3 = LeakyReLU(alpha=0.2)(bn3)
    
    #residual blocks
    rb=make_basic_block_layer(filter_num=256, blocks=res_num)(lr3)
    
    #decoder
    rs1=Resizing(512,512)(rb)
    dconv1=Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(rs1)
    bn4=BatchNormalization()(dconv1)
    lr4=LeakyReLU(alpha=0.2)(bn4)
    
    rs2=Resizing(512,512)(lr4)
    dconv2=Conv2DTranspose(64, (3,3), strides=(1,1), padding='same')(rs2)
    bn5=BatchNormalization()(dconv2)
    lr5=LeakyReLU(alpha=0.2)(bn5)

    
    merge1=add([lr1,lr5])
    
    
    rs3=Resizing(512,512)(merge1)
    dconv3=Conv2DTranspose(1, (3,3), strides=(1,1), padding='same')(rs3)
    bn6=BatchNormalization()(dconv3)
    lr6=LeakyReLU(alpha=0.2)(bn6)
    
    outp=add([inp,lr6])
    
    model = Model(inputs = inp, outputs = outp)
    model.summary()
    
    return model
    
        




'''
dsnms=os.listdir(dspath)
imgnms=[]
kvp=[]
current=[]
exposure=[]
ctdi=[]
sloc=[]
pos=[]
#dlp=[]


for i in range(len(dsnms)):
    
    ds=dicom.dcmread(os.path.join(dspath,dsnms[i]))
    if "PixelData" in ds and ds.pixel_array.shape==(512,512):
        #img=set_window(ds.pixel_array,ds,"midastinum")
        #plt.imsave(os.path.join(imgpath,dsnms[i]+".png"),img,cmap="gray")
        kvp.append(ds.KVP)
        current.append(ds.XRayTubeCurrent)
        exposure.append(ds.Exposure)
        ctdi.append(ds.CTDIvol)
        #dpl.append
        sloc.append(ds.SliceLocation)
        pos.append(ds.InStackPositionNumber)
        imgnms.append(dsnms[i])

df = pd.DataFrame(list(zip(imgnms,kvp,current,exposure,ctdi,sloc,pos)),
                  columns =['img','kvp', 'current','exposure','CTDI','slice location','position index'])
df.to_csv(csvpath+"paras_TestLD.csv",sep=" ") 

'''
    

'''
patoolib.extract_archive("IMAGES.rar", outdir="/IMAGES/")
'''



dspath="/home/lxia/LDSimulation/LowDoseSimulation/IMAGES/"
csvpath="/home/lxia/LDSimulation/LowDoseSimulation/paras_TestLD.csv"
  
       
df=pd.read_csv(csvpath,sep=' ')
df120=df.loc[df['kvp']==120.0].sort_values(by=['slice location'])
df120=df120.loc[df120['slice location'] >= 90.0]
df80=df.loc[df['kvp']==80.0].sort_values(by=['slice location'])


imgs120_train,imgs80_train=readimg(df120,df80,dspath,"midastinum",0.8)
imgs120_test,imgs80_test=readimg(df120,df80,dspath,"midastinum",0.2)


#8 residual blocks
model = generator(8)
model.compile(optimizer=Adam(learning_rate=0.01), loss=MeanAbsoluteError(), metrics=['accuracy'])

csvlogger_path = "/home/lxia/LDSimulation/DL/csvlogger/"
csv_logger = CSVLogger(csvlogger_path+'model_rb8_l1_adam001_training.csv')

checkpoint_path = "/home/lxia/LDSimulation/DL/checkpoint/"
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

filepath = checkpoint_path+"model_rb8_l1_adam001_{epoch:02d}_{accuracy:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath,
                             monitor = 'accuracy',
                             verbose = 1,
                             save_best_only = True,
                             save_weights_only = True,
                             mode = 'auto',
                             save_freq = 'epoch',
                             options = None)

early = EarlyStopping(monitor='accuracy',
                      min_delta = 0,
                      patience = 100,
                      verbose = 1,
                      mode = 'auto')



hist = model.fit(imgs120_train,
                  imgs80_train,
                  batch_size = 2,
                  epochs = 1000,
                  verbose = 1,
                  callbacks = [checkpoint,early,csv_logger]
                  )



performance_path= "/home/lxia/LDSimulation/DL/performance/"
if not os.path.exists(performance_path):
    os.makedirs(performance_path)
    
plt.plot(hist.history['accuracy'])
plt.title('accuracy curve, 2 skip connections, 8 residual blocks, adam (0.01), MAE')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig(performance_path+"accuracy curve_8rb_adam001_mae.png")
plt.show()

plt.plot(hist.history['loss'])
plt.title('loss curve, 8 residual blocks, adam (0.01), MAE')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig(performance_path+"loss curve_8rb_adam001_mae.png")
plt.show()



imgs120_test20=imgs120_test[20]
plt.figure(figsize = (20,20))
plt.imsave('test_8rb_adam001_l1_ep1000.png',imgs120_test20,cmap="gray")

x=np.expand_dims(imgs120_test20,axis=0)
output=model.predict(x)

plt.figure(figsize = (20,20))
plt.imsave('output_8rb_adam001_l1_ep1000.png',np.squeeze(output[0],axis=(2,)),cmap="gray")

plt.figure(figsize = (20,20))
plt.imsave('output_diff_8rb_adam001_l1_ep1000.png',np.squeeze(output[0],axis=(2,))-imgs120_test20,cmap="gray")

imgs80_test20=imgs80_test[20]
plt.figure(figsize = (20,20))
plt.imsave('goal_8rb_adam001_l1_ep1000.png',imgs80_test20,cmap="gray")

plt.figure(figsize = (20,20))
plt.imshow('goal_diff_8rb_adam001_l1_ep1000.png',imgs120_test20-imgs80_test20,cmap="gray")
