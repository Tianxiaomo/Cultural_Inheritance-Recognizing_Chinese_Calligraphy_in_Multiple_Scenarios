# coding=utf-8
from keras import Input, Model
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.layers import Concatenate, Conv2D, UpSampling2D, BatchNormalization,add

import cv2
import numpy as np

import cfg

"""
input_shape=(img.height, img.width, 3), height and width must scaled by 32.
So images's height and width need to be pre-processed to the nearest num that
scaled by 32.And the annotations xy need to be scaled by the same ratio 
as height and width respectively.
"""
import tensorflow as tf
from keras import backend as K
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


class East:

    def __init__(self):
        self.input_img = Input(name='input_img',
                               shape=(None, None, cfg.num_channels),
                               dtype='float32')
        vgg16 = VGG16(input_tensor=self.input_img,
                      # weights='imagenet',
                      include_top=False)
        if cfg.locked_layers:
            # locked first two conv layers
            locked_layers = [vgg16.get_layer('block1_conv1'),
                             vgg16.get_layer('block1_conv2')]
            for layer in locked_layers:
                layer.trainable = False
        self.f = [vgg16.get_layer('block%d_pool' % i).output
                  for i in cfg.feature_layers_range]
        self.f.insert(0, None)
        self.diff = cfg.feature_layers_range[0] - cfg.feature_layers_num

    def g(self, i):
        # i+diff in cfg.feature_layers_range
        assert i + self.diff in cfg.feature_layers_range, \
            ('i=%d+diff=%d not in ' % (i, self.diff)) + \
            str(cfg.feature_layers_range)
        if i == cfg.feature_layers_num:
            bn = BatchNormalization()(self.h(i))
            return Conv2D(32, 3, activation='relu', padding='same')(bn)
        else:
            return UpSampling2D((2, 2))(self.h(i))

    def h(self, i):
        # i+diff in cfg.feature_layers_range
        assert i + self.diff in cfg.feature_layers_range, \
            ('i=%d+diff=%d not in ' % (i, self.diff)) + \
            str(cfg.feature_layers_range)
        if i == 1:
            return self.f[i]
        else:
            concat = Concatenate(axis=-1)([self.g(i - 1), self.f[i]])
            bn1 = BatchNormalization()(concat)
            conv_1 = Conv2D(128 // 2 ** (i - 2), 1,
                            activation='relu', padding='same',)(bn1)
            bn2 = BatchNormalization()(conv_1)
            conv_3 = Conv2D(128 // 2 ** (i - 2), 3,
                            activation='relu', padding='same',)(bn2)
            return conv_3

    def east_network(self):
        before_output = self.g(cfg.feature_layers_num)
        inside_score = Conv2D(1, 1, padding='same', name='inside_score'
                              )(before_output)
        side_v_code = Conv2D(2, 1, padding='same', name='side_vertex_code'
                             )(before_output)
        side_v_coord = Conv2D(4, 1, padding='same', name='side_vertex_coord'
                              )(before_output)
        east_detect = Concatenate(axis=-1,
                                  name='east_detect')([inside_score,
                                                       side_v_code,
                                                       side_v_coord])
        return Model(inputs=self.input_img, outputs=east_detect)


from keras.layers.core import Layer

class GetBox(Layer):

    def __init__(self, **kwargs):
        # self.output_dim = output_shape
        super(GetBox, self).__init__(**kwargs)

    def build(self, input_shape):
        None

    def call(self, x):

        _,contours, hierarchy = cv2.findContours(x, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        def draw_min_rect_circle(cnts):
            rects = []
            for cnt in cnts:
                min_rect = cv2.minAreaRect(cnt)
                min_rect = np.int0(cv2.boxPoints(min_rect))
                rects.append(min_rect)
            return rects
        return draw_min_rect_circle(contours)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],4,None)


class MyEast:

    def __init__(self):
        self.input_img = Input(name='input_img',
                               shape=(None, None, cfg.num_channels),
                               dtype='float32')
        vgg16 = VGG16(input_tensor=self.input_img,
                      # weights='imagenet',
                      include_top=False)
        if cfg.locked_layers:
            # locked first two conv layers
            locked_layers = [vgg16.get_layer('block1_conv1'),
                             vgg16.get_layer('block1_conv2')]
            for layer in locked_layers:
                layer.trainable = False
        self.f = [vgg16.get_layer('block%d_pool' % i).output
                  for i in cfg.feature_layers_range]
        self.f.insert(0, None)
        self.diff = cfg.feature_layers_range[0] - cfg.feature_layers_num

    def g(self, i):
        # i+diff in cfg.feature_layers_range
        assert i + self.diff in cfg.feature_layers_range, \
            ('i=%d+diff=%d not in ' % (i, self.diff)) + \
            str(cfg.feature_layers_range)
        if i == cfg.feature_layers_num:
            bn = BatchNormalization()(self.h(i))
            return Conv2D(32, 3, activation='relu', padding='same')(bn)
        else:
            return UpSampling2D((2, 2))(self.h(i))

    def h(self, i):
        # i+diff in cfg.feature_layers_range
        assert i + self.diff in cfg.feature_layers_range, \
            ('i=%d+diff=%d not in ' % (i, self.diff)) + \
            str(cfg.feature_layers_range)
        if i == 1:
            return self.f[i]
        else:
            concat = Concatenate(axis=-1)([self.g(i - 1), self.f[i]])
            bn1 = BatchNormalization()(concat)
            conv_1 = Conv2D(128 // 2 ** (i - 2), 1,
                            activation='relu', padding='same',)(bn1)
            bn2 = BatchNormalization()(conv_1)
            conv_3 = Conv2D(128 // 2 ** (i - 2), 3,
                            activation='relu', padding='same',)(bn2)
            return conv_3

    def east_network(self):
        before_output = self.g(cfg.feature_layers_num)
        inside_score = Conv2D(1, 1, padding='same', name='inside_score'
                              )(before_output)
        side_v_code = Conv2D(2, 1, padding='same', name='side_vertex_code'
                             )(before_output)
        side_v_coord = Conv2D(4, 1, padding='same', name='side_vertex_coord'
                              )(before_output)

        box = GetBox()(inside_score)

        east_detect = Concatenate(axis=-1,
                                  name='east_detect')([inside_score,
                                                       side_v_code,
                                                       side_v_coord,
                                                       box])
        return Model(inputs=self.input_img, outputs=east_detect)

    def get_box(self,score):

        # tf.findContours()

        _,contours, hierarchy = cv2.findContours(score, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        def draw_min_rect_circle(cnts):
            rects = []
            for cnt in cnts:
                min_rect = cv2.minAreaRect(cnt)
                min_rect = np.int0(cv2.boxPoints(min_rect))
                rects.append(min_rect)
            return rects

        return draw_min_rect_circle(contours)


class ResEast:

    def __init__(self):
        self.input_img = Input(name='input_img',
                               shape=(None, None, cfg.num_channels),
                               dtype='float32')
        Res = ResNet50(input_tensor=self.input_img,
                      # weights='imagenet',
                      include_top=False)
        if cfg.locked_layers:
            # locked first two conv layers
            locked_layers = [Res.get_layer('block1_conv1'),
                             Res.get_layer('block1_conv2')]
            for layer in locked_layers:
                layer.trainable = False
        self.f = [Res.get_layer('block%d_pool' % i).output
                  for i in cfg.feature_layers_range]
        self.f.insert(0, None)
        self.diff = cfg.feature_layers_range[0] - cfg.feature_layers_num

    def g(self, i):
        # i+diff in cfg.feature_layers_range
        assert i + self.diff in cfg.feature_layers_range, \
            ('i=%d+diff=%d not in ' % (i, self.diff)) + \
            str(cfg.feature_layers_range)
        if i == cfg.feature_layers_num:
            bn = BatchNormalization()(self.h(i))
            return Conv2D(32, 3, activation='relu', padding='same')(bn)
        else:
            return UpSampling2D((2, 2))(self.h(i))

    def h(self, i):
        # i+diff in cfg.feature_layers_range
        assert i + self.diff in cfg.feature_layers_range, \
            ('i=%d+diff=%d not in ' % (i, self.diff)) + \
            str(cfg.feature_layers_range)
        if i == 1:
            return self.f[i]
        else:
            concat = Concatenate(axis=-1)([self.g(i - 1), self.f[i]])
            bn1 = BatchNormalization()(concat)
            conv_1 = Conv2D(128 // 2 ** (i - 2), 1,
                            activation='relu', padding='same',)(bn1)
            bn2 = BatchNormalization()(conv_1)
            conv_3 = Conv2D(128 // 2 ** (i - 2), 3,
                            activation='relu', padding='same',)(bn2)
            return conv_3

    def east_network(self):
        before_output = self.g(cfg.feature_layers_num)
        inside_score = Conv2D(1, 1, padding='same', name='inside_score'
                              )(before_output)
        side_v_code = Conv2D(2, 1, padding='same', name='side_vertex_code'
                             )(before_output)
        side_v_coord = Conv2D(4, 1, padding='same', name='side_vertex_coord'
                              )(before_output)
        east_detect = Concatenate(axis=-1,
                                  name='east_detect')([inside_score,
                                                       side_v_code,
                                                       side_v_coord])
        return Model(inputs=self.input_img, outputs=east_detect)


from keras.engine.topology import Layer
import tensorflow as tf

class Resize(Layer):

    def __init__(self, output_shape, **kwargs):
        self.output_dim = output_shape
        super(Resize, self).__init__(**kwargs)

    def build(self, input_shape):
        None

    def call(self, x):
        return tf.image.resize_images(x,self.output_dim)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim[0],self.output_dim[1],input_shape[-1])


class PixelLink:

    def __init__(self):
        self.input_img = Input(name='input_img',
                               shape=(640,640,cfg.num_channels),
                               dtype='float32')
        vgg16 = VGG16(input_tensor=self.input_img,
                      # weights=None,
                      include_top=False)
        if cfg.locked_layers:
            # locked first two conv layers
            locked_layers = [vgg16.get_layer('block1_conv1'),
                             vgg16.get_layer('block1_conv2')]
            for layer in locked_layers:
                layer.trainable = False
        self.f = []
        self.f.append(vgg16.get_layer('block2_conv2').output)
        self.f.append(vgg16.get_layer('block3_conv3').output)
        self.f.append(vgg16.get_layer('block4_conv3').output)
        self.f.append(vgg16.get_layer('block5_conv3').output)
        self.f.insert(0, None)
        self.f.insert(0, None)

    def upsample(self,x,target,init=False):
        target_shape = list(target._keras_shape[1:-1])
        upscored = Resize(target_shape)(x)
        if init:
            return upscored
        else:
            return add([upscored,target])

    def east_network(self):
        block2 = self.f[2]     # 1/2
        block3 = self.f[3]     # 1/4
        block4 = self.f[4]     # 1/8
        block5 = self.f[5]     # 1/16

        fc1 = Conv2D(512, (3, 3),activation='relu',padding='same',name='fc1')(block5)
        fc2 = Conv2D(512, (3, 3),activation='relu',padding='same',name='fc2')(fc1)

        # -------------------------------------------------------------#

        link1 = Conv2D(1, (1, 1),activation='relu',padding='same',name='link1')(fc2)

        link2 = Conv2D(1, (1, 1),activation='relu',padding='same',name='link2')(block5)
        link = self.upsample(link1,link2,init=True)        # 1/16

        link3 = Conv2D(1, (1, 1),activation='relu',padding='same',name='link3')(block4)
        link = self.upsample(link,link3)  # 1/8

        link4 = Conv2D(1, (1, 1),activation='relu',padding='same',name='link4')(block3)
        link = self.upsample(link,link4)   # 1/4

        #-------------------------------------------------------------#

        cls1 = Conv2D(2, (1, 1), activation='relu', padding='same', name='cls1')(fc2)

        cls2 = Conv2D(2, (1, 1), activation='relu', padding='same', name='cls2')(block5)
        cls = self.upsample(cls1,cls2,init=True)

        cls3 = Conv2D(2, (1, 1), activation='relu', padding='same', name='cls3')(block4)
        cls = self.upsample(cls,cls3)

        cls4 = Conv2D(2, (1, 1), activation='relu', padding='same', name='cls4')(block3)
        cls = self.upsample(cls,cls4)

        # -------------------------------------------------------------#

        coord1 = Conv2D(4, (1, 1), activation='relu', padding='same', name='coord1')(fc2)

        coord2 = Conv2D(4, (1, 1), activation='relu', padding='same', name='coord2')(block5)
        coord = self.upsample(coord1,coord2,init=True)

        coord3 = Conv2D(4, (1, 1), activation='relu', padding='same', name='coord3')(block4)
        coord = self.upsample(coord,coord3)

        coord4 = Conv2D(4, (1, 1), activation='relu', padding='same', name='coord4')(block3)
        coord = self.upsample(coord,coord4)

        pixel_detect = Concatenate(axis=-1,name='pixellink_detect')([cls,link,coord])
        return Model(inputs=self.input_img, outputs=pixel_detect)


if __name__ == '__main__':
    east = MyEast()
    east_network = east.east_network()
    east_network.summary()
