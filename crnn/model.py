#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: 
@file: model.py
@time: 2019/1/7 15:59
@desc:
'''
from keras import layers
from keras import backend
from keras.backend import transpose
from keras.layers import *
# from keras.backend impor
from keras import Model
from keras.applications import DenseNet121
import tensorflow as tf

import cfg

class CRNN_slim:
    '''
    CRNN : CNN(vgg16) + RNN(BLSTM) + CTC
    '''
    def __init__(self,
                 weight=None,
                 num=cfg.num_classes,
                 img_w=cfg.img_w,
                 img_h=cfg.img_h,
                 img_c=cfg.img_c):

        self.input_img = Input(name='input_image',shape=(img_h,img_w,img_c),dtype='float32')
        self.num = num
        model = self.build_model()
        if weight:
            print(weight)
            model.load_weights(weight)
        self.model = model

    def build_model(self):

        x = Conv2D(64, (3, 3),activation='relu',padding='same',name='block1_conv1')(self.input_img)
        x = MaxPooling2D(2, strides=2, name='block1_pool')(x)          # 64x16x64

        x = Conv2D(64, (3, 3),activation='relu',padding='same',name='block2_conv1')(x)
        x = MaxPooling2D(2, strides=2, name='block2_pool')(x)           # 128x8x32

        x = Conv2D(128, (3, 3),activation='relu',padding='same',name='block3_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3),activation='relu',padding='same',name='block3_conv2')(x)
        x = MaxPooling2D(2, strides=(2,1), name='block3_pool')(x)
        x = convolutional.ZeroPadding2D(padding=(0, 1))(x)              # 256x4x16

        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = MaxPooling2D(2, strides=(2,1), name='block4_pool')(x)
        x = convolutional.ZeroPadding2D(padding=(0, 1))(x)              # 512x2x42

        x = Conv2D(256, (2, 2), activation='relu',name='block5_conv1')(x)
        x = BatchNormalization()(x)                                     # 512x1x41

        bn,h,w,c = x.get_shape().as_list()

        x = Reshape((w,c),input_shape=(h,w,c))(x)

        #==============================================RNN===============================================#
        x = Bidirectional(LSTM(256,input_shape=(None, 256),return_sequences=True),name='bilstm_1')(x)
        x = Dense(256,activation='linear',name='fc_1')(x)
        x = Bidirectional(LSTM(256,input_shape=(None, 256),return_sequences=True,name='bilstm_2'))(x)
        x = Dense(self.num,activation='softmax',name='output')(x)

        return Model(inputs=self.input_img,outputs=x)

if __name__ == '__main__':
    # crnn = CRNN(160,32,3).build_model()
    # crnn.summary()
    import tensorflow as tf
    num_cores = 1
    GPU = 0
    CPU = 1
    if GPU:
        num_GPU = 1
        num_CPU = 1
    if CPU:
        num_CPU = 1
        num_GPU = 0

    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
            inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
            device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
    session = tf.Session(config=config)
    K.set_session(session)

    crnn_my = CRNN_slim().build_model()
    crnn_my.summary()
