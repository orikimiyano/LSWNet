from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from keras import backend as K
import tensorflow as tf

label_1 = [0, 0, 0]

label_2 = [128, 0, 0]
label_3 = [0,128,0]
label_4 = [128,128,0]
label_5 = [0,0,128]
label_6 = [128,0,128]
label_7 = [0,128,128]
label_8 = [128,128,128]
label_9 = [64,0,0]
label_10 = [192,0,0]
label_11 = [64, 128, 0]

label_12 = [192,128,0]
label_13 = [64,0,128]
label_14 = [192,0,128]
label_15 = [64,128,128]
label_16 = [192, 128, 128]
label_17 = [0, 64, 0]
label_18 = [128, 64, 0]
label_19 = [0, 192, 0]
label_20 = [128, 192, 0]


#COLOR_DICT = np.array([label_1, label_2, label_12, label_8])


COLOR_DICT = np.array([label_1, label_2, label_2, label_4, label_12, label_6, label_8, label_8, label_9, label_10, label_11, label_2, label_12, label_14, label_15, label_16, label_17, label_18, label_19, label_20])


#COLOR_DICT = np.array([label_1, label_2, label_3, label_4, label_5, label_6, label_7, label_8, label_9, label_10, label_11, label_12, label_13, label_14, label_15, label_16, label_17, label_18, label_19, label_20])

IMAGE_SIZE = 256

def adjustData(img,label,flag_multi_class,num_class):
    if (flag_multi_class):
        img = img/255.
        label = label[:,:,:,0] if (len(label.shape)==4) else label[:,:,0]
        new_label = np.zeros(label.shape+(num_class,))
        for i in range(num_class):
            new_label[label==i,i] = 1
        label = new_label
    elif (np.max(img)>1):
        img = img/255.
        label = label/255.
        label[label>0.5] = 1
        label[label<=0.5] = 0
    return (img,label)

def trainGenerator(batch_size,aug_dict,train_path,image_folder,label_folder,image_color_mode='rgb',
                   label_color_mode='rgb',image_save_prefix='image',label_save_prefix='label',
                   flag_multi_class=True,num_class=20,save_to_dir=None,target_size=(IMAGE_SIZE,IMAGE_SIZE),seed=1):
    image_datagen = ImageDataGenerator(**aug_dict)
    label_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix = image_save_prefix,
        seed = seed
    )
    label_generator = label_datagen.flow_from_directory(
        train_path,
        classes = [label_folder],
        class_mode = None,
        color_mode = label_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix = label_save_prefix,
        seed = seed
    )
    train_generator = zip(image_generator,label_generator)
    for img,label in train_generator:
        img,label = adjustData(img,label,flag_multi_class,num_class)
        yield img,label


def getFileNum(test_path):
    for root, dirs, files in os.walk(test_path):
        lens=len(files)
        return lens

def testGenerator(test_path,target_size = (IMAGE_SIZE,IMAGE_SIZE),flag_multi_class=True,as_gray=False):
    num_image = getFileNum(test_path)
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        #img = np.expand_dims(img,3)
        yield img

def saveResult(save_path,npyfile,flag_multi_class=True):
    for i,item in enumerate(npyfile):
        if flag_multi_class:
            img = item
            img_out = np.zeros(img[:, :, 0].shape + (3,))
            for row in range(img.shape[0]):
                for col in range(img.shape[1]):
                    index_of_class = np.argmax(img[row, col])
                    img_out[row, col] = COLOR_DICT[index_of_class]
                    #img_out[row, col] = index_of_class
            img = img_out.astype(np.uint8)
            io.imsave(os.path.join(save_path, '%s_predict.png' % i), img)
        else:
            img = item[:, :, 0]
            img[img > 0.5] = 1
            img[img <= 0.5] = 0
            img = img * 255.
            io.imsave(os.path.join(save_path, '%s_predict.png' % i), img)
