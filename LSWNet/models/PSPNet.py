#coding=utf-8
from keras.models import *
from keras.layers import *
import keras.backend as K
import tensorflow as tf
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from models.loss import *

IMAGE_SIZE = 384

def pool_block(inp, pool_factor):
    h = K.int_shape(inp)[1]
    w = K.int_shape(inp)[2]
    pool_size = strides = [int(np.round( float(h) / pool_factor)), int(np.round( float(w)/ pool_factor))]
    x = AveragePooling2D(pool_size, strides=strides, padding='same')(inp)
    x = Conv2D(256, (1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1])*strides[0], int(x.shape[2])*strides[1])))(x)
    x = Conv2D(256, (1, 1), padding='same', activation='relu')(x)
    return x

def net(input_size=(IMAGE_SIZE, IMAGE_SIZE, 3), num_class=20):
    assert IMAGE_SIZE % 192 == 0

    img_input = Input(input_size)

    x = (Conv2D(64, (3, 3), activation='relu', padding='same'))(img_input)
    x = (BatchNormalization())(x)
    x = (MaxPooling2D((2, 2)))(x)
    f1 = x
    # 192 x 192

    x = (Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
    x = (BatchNormalization())(x)
    x = (MaxPooling2D((2, 2)))(x)
    f2 = x
    # 96 x 96
    x = (Conv2D(256, (3, 3), activation='relu', padding='same'))(x)
    x = (BatchNormalization())(x)
    x = (MaxPooling2D((2, 2)))(x)
    f3 = x
    # 48 x 48
    x = (Conv2D(256, (3, 3), activation='relu', padding='same'))(x)
    x = (BatchNormalization())(x)
    x = (MaxPooling2D((2, 2)))(x)
    f4 = x

    # 24 x 24
    o = f4
    pool_factors = [1, 2, 3, 6]
    pool_outs = [o]
    for p in pool_factors:
        pooled = pool_block(o, p)
        pool_outs.append(pooled)

    o = Concatenate(axis=3)(pool_outs)
    o = Conv2D(384, (3, 3), activation='relu', padding='same')(o)
    o = BatchNormalization()(o)

    o = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1])*8, int(x.shape[2])*8)))(x)
    o = (UpSampling2D(size=(2, 2)))(o)
    o = Conv2D(num_class, (1, 1), padding='same', activation='softmax')(o)

    model = Model(input=img_input, output=o)

    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

