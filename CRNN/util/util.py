#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: 
@file: util.py
@time: 2019/1/9 14:42
@desc:
'''
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os
import time

def getTimeStr():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

def gpuConfig(gpu_num):
    import tensorflow as tf
    import keras.backend.tensorflow_backend as KTF
    import os
    if gpu_num is not None:
        if isinstance(gpu_num, (list, tuple)):
            gpu_num = ','.join(str(i) for i in gpu_num)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        sess = tf.Session(config=config)
        KTF.set_session(sess)
        print('GPU config done!')
    else:
        print('Use CPU!')
# if cfg.gpu != None:
# #     gpuConfig(cfg.gpu)