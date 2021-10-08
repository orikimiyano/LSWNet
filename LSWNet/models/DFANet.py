from keras import *
from keras.layers import *
import numpy as np
import glob
import cv2
import os
from keras.models import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

IMAGE_SIZE = 256

def depthsep2d(f,conv):
    
    x = DepthwiseConv2D(3, padding='same', use_bias=False)(conv)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(f, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x
        
def enc1(tensor,F = [12,12,48],times=4):
    
    for i in range(times):
        
        conv = depthsep2d(F[0],tensor)
        conv = depthsep2d(F[1],conv)
        conv = depthsep2d(F[2],conv)
        tensor = conv
        
    conv = Conv2D(F[2],(1,1),strides=2,activation = 'relu',padding = 'same')(conv)
    
    return conv
   
def attention_mod(conv):
    
    x = GlobalAveragePooling2D()(conv)
    x = Dense(1000, activation='relu')(x)
    x = Dense(192, activation='sigmoid')(x)
    x = Reshape((1,1,192))(x)
    fc = multiply([x,conv])

    
    return fc


def decode_conv(conv,f=48):
    
    conv = Convolution2D(f, (1, 1), padding='same')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    
    return conv
    

    
def conv_cls(conv,num_classes = 5):
    
    conv = Convolution2D(num_classes, (3,3), padding='same',activation = 'softmax')(conv)

    
    return conv


def net(input_size=(IMAGE_SIZE, IMAGE_SIZE, 3), num_class=20):

    input_map= Input(input_size)

    conv1 = Conv2D(8,(3,3),strides=1,activation = 'relu',padding = 'same')(input_map)

    enc1_2 = enc1(conv1,F = [12,12,48],times=3)

    enc1_3 = enc1(enc1_2,F = [24,24,96],times=6)

    enc1_4 = enc1(enc1_3,F = [48,48,192],times=3)

    fc1 = attention_mod(enc1_4)

    #=====================================
    conv_d1 = UpSampling2D(size=(4,4))(fc1)  

    c1 = concatenate([enc1_2,conv_d1],axis=3)

    enc2_2 = enc1(c1,F = [12,12,48],times=3)

    c2 = concatenate([enc2_2,enc1_3],axis=3)

    enc2_3 = enc1(c2,F = [24,24,96],times=3)

    c3 = concatenate([enc2_3,enc1_4],axis=3)

    enc2_4 = enc1(c3,F = [48,48,192],times=3)

    fc2 = attention_mod(enc2_4)
    #================================

    conv_d2 = UpSampling2D(size=(4,4))(fc2)

    c2_1 = concatenate([enc2_2,conv_d2],axis=3)

    enc3_2 = enc1(c2_1,F = [12,12,48],times=3)

    c2_2 = concatenate([enc2_3,enc3_2],axis=3)

    enc3_3 = enc1(c2_2,F = [24,24,96],times=3)

    c2_3 = concatenate([enc2_4,enc3_3],axis=3)

    enc3_4 = enc1(c2_3,F = [48,48,192],times=3)

    fc3 = attention_mod(enc3_4)

    #=======================================
    ## decoder section of paper

    ## from attention section 

    d1 = UpSampling2D(size=(4,4))(fc1)
    d1 = decode_conv(d1,f=48)

    d2 = UpSampling2D(size=(8,8))(fc2)
    d2 = decode_conv(d2,f=48)

    d3 = UpSampling2D(size=(16,16))(fc3)
    d3 = decode_conv(d3,f=48)



    ## from encoder section 
    d1_1  = UpSampling2D(size=(1,1))(enc1_2)
    d1_1 = decode_conv(d1_1,f=192)

    d2_1  = UpSampling2D(size=(2,2))(enc2_2)
    d2_1 = decode_conv(d2_1,f=192)

    d3_1 = UpSampling2D(size=(4,4))(enc3_2)
    d3_1 = decode_conv(d3_1,f=192)


    shallow = Add()([d1,d2,d3])
    shallow = decode_conv(shallow,f=192)


    final = Add()([shallow,d1_1,d2_1,d3_1])
    final = decode_conv(final,f=192)


    output = UpSampling2D(size=(2,2))(conv_cls(final,num_classes = num_class))

    model = Model(inputs = input_map,outputs = output)

    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    return model


