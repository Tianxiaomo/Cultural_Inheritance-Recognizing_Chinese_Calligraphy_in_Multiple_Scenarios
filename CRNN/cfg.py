#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: 
@file: cfg.py
@time: 2019/1/7 15:56
@desc:
'''
import time
def getTimeStr():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


model = 'CRNN'
checkpoint = None
char_path = './util/char_gb2312.txt'

model_weights_path = 'checkpoints/weights_%s.{epoch:03d}-{val_loss:.3f}-{val_acc:.3f}.h5' % model
tensorboard_dir = './logs/%s_%s' % (model,getTimeStr())

with open(char_path, 'r', encoding='utf-8') as file:
    char_dict = {num: char.strip() for num, char in enumerate(file.readlines())}
    lexicon_len = len(char_dict)

gpu = 3

num_classes = lexicon_len

img_w, img_h,img_c = 640, 32, 3

# data parameter
from easydict import EasyDict as edict
data_para =edict({'img_w':None,
    'img_h':32,
    'img_c':3,
    'vertical':True,
    'background_path':'/img/',
    'font_path':'./fonts/'
})

# Network parameters
train_batch_size = 96
val_batch_size = 128

downsample_factor = 4
max_text_len = 9

locked_layers = False

patience = 6    # earlystop
lr  = 0.0005     #learn rate
train_steps = 10
val_steps = 5
epoch_num = 100

img = 'img1'
