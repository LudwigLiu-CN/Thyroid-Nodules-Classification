# -*- coding: utf-8 -*
from __future__ import print_function
import keras
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D, Convolution2D
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers


def vgg_1(input_shape, classes):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same',input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation('softmax'))
    return model



def vgg_2(input_shape, classes):
    model = Sequential()
    
    model.add(ZeroPadding2D((1,1),input_shape=input_shape))#卷积输入层，指定了输入图像的大小
 
    model.add(Convolution2D(64, 3, 3, activation='relu'))#64个3x3的卷积核，生成64*224*224的图像，激活函数为relu
 
    model.add(ZeroPadding2D((1,1)))
 
    model.add(Convolution2D(64, 3, 3, activation='relu'))#再来一次卷积 生成64*224*224
 
    model.add(MaxPooling2D((2,2), strides=(2,2)))#pooling操作，相当于变成64*112*112
 
 
 
    model.add(ZeroPadding2D((1,1)))
 
    model.add(Convolution2D(128, 3, 3, activation='relu'))
 
    model.add(ZeroPadding2D((1,1)))
 
    model.add(Convolution2D(128, 3, 3, activation='relu'))
 
    model.add(MaxPooling2D((2,2), strides=(2,2)))#128*56*56
 
 
 
    model.add(ZeroPadding2D((1,1)))
 
    model.add(Convolution2D(256, 3, 3, activation='relu'))
 
    model.add(ZeroPadding2D((1,1)))
 
    model.add(Convolution2D(256, 3, 3, activation='relu'))
 
    model.add(ZeroPadding2D((1,1)))
 
    model.add(Convolution2D(256, 3, 3, activation='relu'))
 
    model.add(MaxPooling2D((2,2), strides=(2,2)))#256*28*28
 
 
 
    model.add(ZeroPadding2D((1,1)))
 
    model.add(Convolution2D(512, 3, 3, activation='relu'))
 
    model.add(ZeroPadding2D((1,1)))
 
    model.add(Convolution2D(512, 3, 3, activation='relu'))
 
    model.add(ZeroPadding2D((1,1)))
 
    model.add(Convolution2D(512, 3, 3, activation='relu'))
 
    model.add(MaxPooling2D((2,2), strides=(2,2)))#512*14*14
 
 
 
    model.add(ZeroPadding2D((1,1)))
 
    model.add(Convolution2D(512, 3, 3, activation='relu'))
 
    model.add(ZeroPadding2D((1,1)))
 
    model.add(Convolution2D(512, 3, 3, activation='relu'))
 
    model.add(ZeroPadding2D((1,1)))
 
    model.add(Convolution2D(512, 3, 3, activation='relu'))
 
    model.add(MaxPooling2D((2,2), strides=(2,2)))  
 
 
 
    model.add(Flatten())
 
    model.add(Dense(4096, activation='relu'))
 
    model.add(Dropout(0.5))
 
    model.add(Dense(4096, activation='relu'))
 
    model.add(Dropout(0.5))
 
    model.add(Dense(classes, activation='softmax'))
 
 
    return model
 