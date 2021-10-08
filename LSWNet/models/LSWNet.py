import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from models.loss import *

IMAGE_SIZE = 256
filter = 32


# filters,kernel_size,strides

def net(pretrained_weights=None, input_size=(IMAGE_SIZE, IMAGE_SIZE, 3), num_class=20):
    inputs = Input(input_size)

    #####-----Hierarchical-Block-1------#####
    # ConvBlock_1_64
    conv1 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=0.3)(conv1)

    conv1 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv1)
    conv1 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    # shutcut1 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(inputs)

    merge1 = concatenate([conv1, inputs], axis=3)
    conv1 = LeakyReLU(alpha=0.3)(merge1)

    # ConvBlock_2_64
    conv2 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=0.3)(conv2)

    conv2 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv2)
    conv2 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)

    # shutcut2 = Conv2D(filter, 1,  padding='same', kernel_initializer='he_normal')(merge1)
    merge2 = concatenate([conv2, conv1], axis=3)
    conv2 = LeakyReLU(alpha=0.3)(merge2)

    # AggregationBlock_3_64
    merge_unit1 = concatenate([conv1, conv2], axis=3)
    conv_root3 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(merge_unit1)
    Aggre3 = LeakyReLU(alpha=0.3)(conv_root3)

    #####-----Hierarchical-Block-1------#####

    pool3 = MaxPool2D(pool_size=(2, 2))(Aggre3)

    #####-----Hierarchical-Block-2------#####
    # ConvBlock_4_128
    conv4 = Conv2D(filter * 2, 3, padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=0.3)(conv4)

    conv4 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv4)
    conv4 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)

    # shutcut1 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(inputs)

    merge4 = concatenate([conv4, pool3], axis=3)
    conv4 = LeakyReLU(alpha=0.3)(merge4)

    # ConvBlock_5_128
    conv5 = Conv2D(filter * 2, 3, padding='same', kernel_initializer='he_normal')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=0.3)(conv5)

    conv5 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv5)
    conv5 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)

    # shutcut2 = Conv2D(filter, 1,  padding='same', kernel_initializer='he_normal')(merge1)
    merge5 = concatenate([conv4, conv5], axis=3)
    conv5 = LeakyReLU(alpha=0.3)(merge5)

    # AggregationBlock_6_128
    merge_unit2 = concatenate([conv4, conv5], axis=3)
    conv_root6 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(merge_unit2)
    Aggre6 = LeakyReLU(alpha=0.3)(conv_root6)

    #####-----Hierarchical-Block-2------#####

    pool6 = MaxPool2D(pool_size=(2, 2))(Aggre6)

    #####-----Hierarchical-Block-3------#####

    # ConvBlock_7_256
    conv7 = Conv2D(filter * 4, 3, padding='same', kernel_initializer='he_normal')(pool6)
    conv7 = BatchNormalization()(conv7)
    conv7 = LeakyReLU(alpha=0.3)(conv7)

    conv7 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv7)
    conv7 = Conv2D(filter * 4, 1, padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    # shutcut1 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(inputs)

    merge7 = concatenate([conv7, pool6], axis=3)
    conv7 = LeakyReLU(alpha=0.3)(merge7)

    # ConvBlock_8_256
    conv8 = Conv2D(filter * 4, 3, padding='same', kernel_initializer='he_normal')(conv7)
    conv8 = BatchNormalization()(conv8)
    conv8 = LeakyReLU(alpha=0.3)(conv8)

    conv8 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv8)
    conv8 = Conv2D(filter * 4, 1, padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    # shutcut2 = Conv2D(filter, 1,  padding='same', kernel_initializer='he_normal')(merge1)
    merge8 = concatenate([conv7, conv8], axis=3)
    conv8 = LeakyReLU(alpha=0.3)(merge8)

    # AggregationBlock_9_256
    merge_unit3 = concatenate([conv7, conv8], axis=3)
    conv_root9 = Conv2D(filter * 4, 1, padding='same', kernel_initializer='he_normal')(merge_unit3)
    Aggre9 = LeakyReLU(alpha=0.3)(conv_root9)

    conv9_1 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(Aggre3)
    conv9_1 = MaxPool2D(pool_size=(2, 2))(conv9_1)
    skip9_1 = concatenate([conv9_1, Aggre6], axis=3)
    conv_skip9 = Conv2D(filter * 4, 1, padding='same', kernel_initializer='he_normal')(skip9_1)
    conv_skip9 = LeakyReLU(alpha=0.3)(conv_skip9)

    conv_skip9 = MaxPool2D(pool_size=(2, 2))(conv_skip9)
    skip9_2 = concatenate([Aggre9, conv_skip9], axis=3)
    conv_skip9 = Conv2D(filter * 4, 1, padding='same', kernel_initializer='he_normal')(skip9_2)
    conv_skip9 = LeakyReLU(alpha=0.3)(conv_skip9)

    #####-----Hierarchical-Block-3------#####

    up9 = UpSampling2D(size=(2, 2))(conv_skip9)

    #####-----Hierarchical-Block-4------#####

    # ConvBlock_10_128
    conv10 = Conv2D(filter * 2, 3, padding='same', kernel_initializer='he_normal')(up9)
    conv10 = BatchNormalization()(conv10)
    conv10 = LeakyReLU(alpha=0.3)(conv10)

    conv10 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv10)
    conv10 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(conv10)
    conv10 = BatchNormalization()(conv10)

    # shutcut1 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(inputs)

    merge10 = concatenate([conv10, up9], axis=3)
    conv10 = LeakyReLU(alpha=0.3)(merge10)

    # ConvBlock_11_128
    conv11 = Conv2D(filter * 2, 3, padding='same', kernel_initializer='he_normal')(conv10)
    conv11 = BatchNormalization()(conv11)
    conv11 = LeakyReLU(alpha=0.3)(conv11)

    conv11 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv11)
    conv11 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(conv11)
    conv11 = BatchNormalization()(conv11)

    # shutcut2 = Conv2D(filter, 1,  padding='same', kernel_initializer='he_normal')(merge1)
    merge11 = concatenate([conv10, conv11], axis=3)
    conv11 = LeakyReLU(alpha=0.3)(merge11)

    # AggregationBlock_12_128
    merge_unit4 = concatenate([conv10, conv11], axis=3)
    conv_root12 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(merge_unit4)
    Aggre12 = LeakyReLU(alpha=0.3)(conv_root12)
    # Aggre12 = Conv2D(filter*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
    #     UpSampling2D(size=(2, 2))(Aggre12))

    con_up12 = UpSampling2D(size=(2, 2))(conv_skip9)
    skip12 = concatenate([Aggre12, con_up12], axis=3)
    conv_skip12 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(skip12)
    conv_skip12 = LeakyReLU(alpha=0.3)(conv_skip12)

    #####-----Hierarchical-Block-4------#####

    up12 = UpSampling2D(size=(2, 2))(conv_skip12)

    #####-----Hierarchical-Block-5------#####

    # ConvBlock_13_64
    conv13 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(up12)
    conv13 = BatchNormalization()(conv13)
    conv13 = LeakyReLU(alpha=0.3)(conv13)

    conv13 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv13)
    conv13 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(conv13)
    conv13 = BatchNormalization()(conv13)

    # shutcut1 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(inputs)

    merge13 = concatenate([conv13, up12], axis=3)
    conv13 = LeakyReLU(alpha=0.3)(merge13)

    # ConvBlock_14_64
    conv14 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(conv13)
    conv14 = BatchNormalization()(conv14)
    conv14 = LeakyReLU(alpha=0.3)(conv14)

    conv14 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv14)
    conv14 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(conv14)
    conv14 = BatchNormalization()(conv14)

    # shutcut2 = Conv2D(filter, 1,  padding='same', kernel_initializer='he_normal')(merge1)
    merge14 = concatenate([conv13, conv14], axis=3)
    conv14 = LeakyReLU(alpha=0.3)(merge14)

    # AggregationBlock_15_64
    merge_unit5 = concatenate([conv13, conv14], axis=3)
    conv_root15 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(merge_unit5)
    Aggre15 = LeakyReLU(alpha=0.3)(conv_root15)

    con_up15 = UpSampling2D(size=(4, 4))(conv_skip9)
    skip15 = concatenate([Aggre15, con_up15], axis=3)
    conv_skip15 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(skip15)
    conv_skip12 = LeakyReLU(alpha=0.3)(conv_skip15)

    #####-----Hierarchical-Block-5------#####

    conv_out = Conv2D(num_class, 1, activation='softmax')(conv_skip12)
    # loss_function = 'categorical_crossentropy'

    model = Model(input=inputs, output=conv_out)

    model.compile(optimizer=Adam(lr=1e-5), loss=[multi_conbination_loss], metrics=['accuracy'])
    model.summary()

    return model
