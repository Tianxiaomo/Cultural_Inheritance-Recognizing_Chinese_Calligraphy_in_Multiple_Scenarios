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


model = 'CRNN_slim'
checkpoint = 'checkpoints/weights_crnn.018-0.161-0.97.h5'
char_path = './util/char_gb2312.txt'

model_weights_path = 'checkpoints/weights_%s.{epoch:03d}-{val_loss:.3f}-{val_acc:.3f}.h5' % model
tensorboard_dir = './logs/%s_%s' % (model,getTimeStr())

with open(char_path, 'r', encoding='utf-8') as file:
    char_dict = {num: char.strip() for num, char in enumerate(file.readlines())}
    lexicon_len = len(char_dict)

gpu = 0

num_classes = lexicon_len

img_w, img_h,img_c = 280, 32, 3

# Network parameters
train_batch_size = 512
val_batch_size = 512

downsample_factor = 4
max_text_len = 9

locked_layers = False

patience = 6    # earlystop
lr  = 0.001     #learn rate
train_steps = 1000
val_steps = 50
epoch_num = 100