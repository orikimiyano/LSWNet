from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
import tensorflow as tf
import math
#from scipy.ndimage import distance_transform_edt as distance
#from keras.metrics import top_k_categorical_accuracy

'''
losses for two-class segmentation 
'''
smooth = 1.


# Cross_entropy_loss

def Cross_entropy_loss(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    crossEntropyLoss = -y_true * tf.log(y_pred)

    return tf.reduce_sum(crossEntropyLoss, -1)


# dice_loss

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def binary_focal_loss(y_true, y_pred) :
    gamma = 2
    alpha = 0.25
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    y_true = tf.cast(y_true, tf.float32)
    alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

    p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
    focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
    return K.mean(focal_loss)


def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def cross_tversky(y_true, y_pred):
    loss_value = Cross_entropy_loss(y_true, y_pred) + tversky_loss(y_true, y_pred)
    return loss_value


def multi_conbination_loss(y_true, y_pred):
    y_true_n = K.reshape(y_true, shape=(-1, 4))
    y_pred_n = K.reshape(y_pred, shape=(-1, 4))
    total_single_loss = 0.
    for i in range(y_pred_n.shape[1]):
        single_loss = cross_tversky(y_true_n[:, i], y_pred_n[:, i])
        total_single_loss += single_loss
    return total_single_loss
