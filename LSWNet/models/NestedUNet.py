#coding=utf-8
import tensorflow as tf
import keras 
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from models.loss import *

IMAGE_SIZE = 256

def conv_block_nested(x, mid_ch, out_ch, kernel_size=3, padding='same'):
    x = Conv2D(mid_ch, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(out_ch, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def net(input_size=(IMAGE_SIZE, IMAGE_SIZE, 3), num_class=20):
     
    inputs = Input(input_size)
    t = 2
    n1 = 64
    filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

    x0_0 = conv_block_nested(inputs, filters[0], filters[0])

    x1_0 = conv_block_nested(MaxPooling2D(strides=2)(x0_0), filters[1], filters[1])
    x0_1 = conv_block_nested(Concatenate()([x0_0, UpSampling2D()(x1_0)]), filters[0], filters[0])

    x2_0 = conv_block_nested(MaxPooling2D(strides=2)(x1_0), filters[2], filters[2])
    x1_1 = conv_block_nested(Concatenate()([x1_0, UpSampling2D()(x2_0)]), filters[1], filters[1])
    x0_2 = conv_block_nested(Concatenate()([x0_0, x0_1, UpSampling2D()(x1_1)]), filters[0], filters[0])

    x3_0 = conv_block_nested(MaxPooling2D(strides=2)(x2_0), filters[3], filters[3])
    x2_1 = conv_block_nested(Concatenate()([x2_0, UpSampling2D()(x3_0)]), filters[2], filters[2])
    x1_2 = conv_block_nested(Concatenate()([x1_0, x1_1, UpSampling2D()(x2_1)]), filters[1], filters[1])
    x0_3 = conv_block_nested(Concatenate()([x0_0, x0_1, x0_2, UpSampling2D()(x1_2)]), filters[0], filters[0])

    x4_0 = conv_block_nested(MaxPooling2D(strides=2)(x3_0), filters[4], filters[4])
    x3_1 = conv_block_nested(Concatenate()([x3_0, UpSampling2D()(x4_0)]), filters[3], filters[3])
    x2_2 = conv_block_nested(Concatenate()([x2_0, x2_1, UpSampling2D()(x3_1)]), filters[2], filters[2])
    x1_3 = conv_block_nested(Concatenate()([x1_0, x1_1, x1_2, UpSampling2D()(x2_2)]), filters[1], filters[1])
    x0_4 = conv_block_nested(Concatenate()([x0_0, x0_1, x0_2, x0_3, UpSampling2D()(x1_3)]), filters[0], filters[0])

    out = Conv2D(num_class, (3, 3), padding='same', activation='softmax')(x0_4)

    # outputHeight = Model(inputs, o).output_shape[1]
    # outputWidth = Model(inputs, o).output_shape[2]
    #
    # out = (Reshape((outputHeight * outputWidth, num_class)))(o)
    # out = Activation('softmax')(out)

    model = Model(input=inputs, output=out)
    # model.outputHeight = outputHeight
    # model.outputWidth = outputWidth

    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

