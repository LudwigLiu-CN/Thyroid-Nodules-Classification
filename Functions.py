# -*- coding: utf-8 -*
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import keras
import resnet
import keras.backend as K
import tensorflow as tf
import FinalNetwork 
import MyNetWork 

from skimage import feature as ft
from skimage import transform
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from keras import backend as K

from scipy import interp  
import matplotlib.pyplot as plt  
from sklearn import svm, datasets  
from sklearn.metrics import roc_curve, auc  
from sklearn.model_selection import StratifiedKFold  


def shuffled_copies(dataSets, label, rl):
    assert len(dataSets[0]) == len(label)
    for i in range(len(dataSets) - 1):
        assert len(dataSets[i]) == len(dataSets[i + 1])

    r = list(range(len(label)))
    random.shuffle(r, lambda: rl)
    p = np.array(r)
    out = []

    for data in dataSets:
        out.append(data[p])
    label = label[p]
    return (tuple(out), label)



def all_features(img):
    hog_features = ft.hog(img,  # input image
                          orientations=8,  # number of bins
                          pixels_per_cell=(20, 20),  # pixel per cell
                          cells_per_block=(2, 2),  # cells per blcok
                          block_norm='L2-Hys',  # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
                          transform_sqrt=True,  # power law compression (also known as gamma correction)
                          feature_vector=True,  # flatten the final vectors
                          visualize=False)  # return HOG map

    temp = img.reshape(128, 128)

    lbp_features = ft.local_binary_pattern(temp,  # input image
                                           P=8,  # Number of circularly symmetric neighbour set points
                                           R=1.0,  # Radius of circle
                                           method='default')  # {'default', 'ror', 'uniform', 'var'}

    haar_features = ft.haar_like_feature(temp,  # input image
                                         0,  # Row-coordinate of top left corner of the detection window.
                                         0,  # Column-coordinate of top left corner of the detection window.
                                         5,  # Width of the detection window.
                                         5,  # Height of the detection window.
                                         feature_type=None  # The type of feature to consider:
                                         )

    _lbp = lbp_features.flatten()

    train = np.concatenate((hog_features, _lbp, haar_features), axis=0)
    return train


def hog(img):
    hog_features = ft.hog(img,  # input image
                          orientations=8,  # number of bins
                          pixels_per_cell=(20, 20),  # pixel per cell
                          cells_per_block=(2, 2),  # cells per blcok
                          block_norm='L2-Hys',  # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
                          transform_sqrt=True,  # power law compression (also known as gamma correction)
                          feature_vector=True,  # flatten the final vectors
                          visualize=False)  # return HOG map
    return hog_features


def lbp(img):
    temp = img.reshape(128, 128)
    lbp_features = ft.local_binary_pattern(temp,  # input image
                                           P=8,  # Number of circularly symmetric neighbour set points
                                           R=1.0,  # Radius of circle
                                           method='default')  # {'default', 'ror', 'uniform', 'var'}
    lbp_features = lbp_features.flatten()
    return lbp_features


def haar(img):
    temp = img.reshape(128, 128)
    haar_features = ft.haar_like_feature(temp,  # input image
                                         0,  # Row-coordinate of top left corner of the detection window.
                                         0,  # Column-coordinate of top left corner of the detection window.
                                         5,  # Width of the detection window.
                                         5,  # Height of the detection window.
                                         feature_type=None  # The type of feature to consider:
                                         )
    return haar_features


def hoglbp(img):
    hog_features = ft.hog(img,  # input image
                          orientations=8,  # number of bins
                          pixels_per_cell=(20, 20),  # pixel per cell
                          cells_per_block=(2, 2),  # cells per blcok
                          block_norm='L2-Hys',  # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
                          transform_sqrt=True,  # power law compression (also known as gamma correction)
                          feature_vector=True,  # flatten the final vectors
                          visualize=False)  # return HOG map

    temp = img.reshape(128, 128)
    lbp_features = ft.local_binary_pattern(temp,  # input image
                                           P=8,  # Number of circularly symmetric neighbour set points
                                           R=1.0,  # Radius of circle
                                           method='default')  # {'default', 'ror', 'uniform', 'var'}
    lbp_features = lbp_features.flatten()

    train = np.concatenate((hog_features, lbp_features), axis=0)
    return train


def hoghaar(img):
    hog_features = ft.hog(img,  # input image
                          orientations=8,  # number of bins
                          pixels_per_cell=(20, 20),  # pixel per cell
                          cells_per_block=(2, 2),  # cells per blcok
                          block_norm='L2-Hys',  # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
                          transform_sqrt=True,  # power law compression (also known as gamma correction)
                          feature_vector=True,  # flatten the final vectors
                          visualize=False)  # return HOG map

    temp = img.reshape(128, 128)
    haar_features = ft.haar_like_feature(temp,  # input image
                                         0,  # Row-coordinate of top left corner of the detection window.
                                         0,  # Column-coordinate of top left corner of the detection window.
                                         5,  # Width of the detection window.
                                         5,  # Height of the detection window.
                                         feature_type=None  # The type of feature to consider:
                                         )

    train = np.concatenate((hog_features, haar_features), axis=0)
    return train


def lbphaar(img):
    temp = img.reshape(128, 128)

    lbp_features = ft.local_binary_pattern(temp,  # input image
                                           P=8,  # Number of circularly symmetric neighbour set points
                                           R=1.0,  # Radius of circle
                                           method='default')  # {'default', 'ror', 'uniform', 'var'}

    haar_features = ft.haar_like_feature(temp,  # input image
                                         0,  # Row-coordinate of top left corner of the detection window.
                                         0,  # Column-coordinate of top left corner of the detection window.
                                         5,  # Width of the detection window.
                                         5,  # Height of the detection window.
                                         feature_type=None  # The type of feature to consider:
                                         )

    _lbp = lbp_features.flatten()

    train = np.concatenate((_lbp, haar_features), axis=0)
    return train



# data:list
def processData(dataSets, label, shuffle, k):
    assert type(dataSets) == list

    if shuffle == True:
        dataSets, label = shuffled_copies(dataSets, label, 0.5)

    length = len(label)
    train_list = []
    val_list = []
    if k == 1:
        x_train = []
        x_val = []
        for data in dataSets:
            temp_train = data[:int(round(length * 0.8))]
            temp_val = data[int(round(length * 0.8)):]
            x_train.append(temp_train)
            x_val.append(temp_val)

        y_train = label[:int(round(length * 0.8))]
        y_val = label[int(round(length * 0.8)):]

        train_list.append([x_train, y_train])
        val_list.append([x_val, y_val])

    else:
        split = 1. / k
        for i in range(k):
            x_train = []
            x_val = []
            for data in dataSets:
                temp_train = np.vstack(
                    (data[:int(round(length * split * i))], data[int(round(length * split * (i + 1))):]))
                temp_val = data[int(round(length * split * i)):int(round(length * split * (i + 1)))]
                x_train.append(temp_train)
                x_val.append(temp_val)

            y_train = np.vstack((label[:int(round(length * split * i))], label[int(round(length * split * (i + 1))):]))
            y_val = label[int(round(length * split * i)):int(round(length * split * (i + 1)))]

            train_list.append([x_train, y_train])
            val_list.append([x_val, y_val])

    return train_list, val_list






def TPR(y_true, y_pred):
    TP = 0.
    P = 0.
    for i in range(y_true.shape[0]):
        if y_true[i][1]==1.:
            P += 1
            if y_pred[i][1]==1.:
                TP += 1
    return TP/P

def TNR(y_true, y_pred):
    TN = 0.
    N = 0.
    for i in range(y_true.shape[0]):
        if y_true[i][0]==1.:
            N += 1
            if y_pred[i][0]==1.:
                TN += 1
    return TN/N



def ROC_AUC(true_label, predict): 
    predict_classes_1=[]
    true_classes_1=[]
    for i in range(predict.shape[0]):
        predict_classes_1.append(predict[i][1])
        true_classes_1.append(true_label[i][1])
    label=np.array(true_classes_1)
    score=np.array(predict_classes_1)

    fpr, tpr, thresholds = roc_curve(label, score, pos_label = 1.)        
    roc_auc = auc(fpr, tpr) 
    plt.plot(fpr, tpr, lw=1) 
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    return fpr, tpr, roc_auc