#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: 
@file: dataset.py
@time: 2019/1/7 22:10
@desc:
'''

import os
import numpy as np
from keras.utils import Sequence

from data_generator.generator import gen_data
import cfg


def data_gen(train=True,batch_size = cfg.train_batch_size,img_w=cfg.img_w,img_h=cfg.img_h,img_c=cfg.img_c):
    # with open(cfg.char_path, 'rb') as file:
    #     char_dict = {num: char.strip().decode('gbk', 'ignore') for num, char in enumerate(file.readlines())}
    #     num_dict = {char:num for num,char in char_dict.items()}

    with open(cfg.char_path, 'r', encoding='utf-8') as file:
        char_dict = {num: char.strip() for num, char in enumerate(file.readlines())}
        num_dict = {char: num for num, char in char_dict.items()}

    maxlabellen = img_w //8
    images = np.zeros([batch_size,img_h,img_w,img_c])
    labels = np.zeros([batch_size,maxlabellen],dtype=np.int32)
    input_length = np.zeros([batch_size,1],dtype=np.int32)
    label_length = np.zeros([batch_size,1])
    while True:
        for i in range(batch_size):
            img,text = gen_data(img_w,img_h)
            img = np.asarray(img)
            images[i] = img
            text = text.replace('；',';')
            text = text.replace('＊', '火')
            text = text.replace('：', ':')
            label = []
            for j in text:
                try:
                    num = num_dict.get(j)
                    if num is None:
                        num = 0
                    label.append(num)
                except Exception:
                    label.append(0)

            label_len = len(label)

            try:
                labels[i,:label_len] = label
            except TypeError:
                print(label)
            input_length[i] = img_w // 4 + 1
            label_length[i] = label_len

        inputs = {'input_image': images,
                'the_labels': labels,
                'input_length': input_length,
                'label_length': label_length,
                }
        outputs = {'ctc': np.zeros([batch_size])}
        yield inputs,outputs


class DATASequence(Sequence):
    def __init__(self,train=True,img_w=cfg.img_w,img_h=cfg.img_h,img_c=cfg.img_c):
        if train == False:
            self.batch_size = cfg.val_batch_size
        else:
            self.batch_size = cfg.train_batch_size
        self.img_w = img_w
        self.img_h = img_h
        self.img_c = img_c

        with open(cfg.char_path, 'r', encoding='utf-8') as file:
            self.char_dict = {num: char.strip() for num, char in enumerate(file.readlines())}
            self.num_dict = {char: num for num, char in self.char_dict.items()}
            self.lexicon_len = len(self.char_dict)

    def __len__(self):
        return self.batch_size

    def __getitem__(self, item):
        maxlabellen = self.img_w // 8
        images = np.zeros([self.batch_size, self.img_h, self.img_w, self.img_c])
        labels = np.zeros([self.batch_size, maxlabellen], dtype=np.int32)
        input_length = np.zeros([self.batch_size, 1], dtype=np.int32)
        label_length = np.zeros([self.batch_size, 1])

        for i in range(self.batch_size):
            img, text = gen_data(self.img_w, self.img_h)
            img = np.asarray(img)
            images[i] = img
            label = []
            for j in text:
                try:
                    num = self.num_dict.get(j)
                    if num is None:
                        num = 0
                    label.append(num)
                except Exception:
                    label.append(0)

            label_len = len(label)

            try:
                labels[i, :label_len] = label
            except TypeError:
                print(label)
            input_length[i] = self.img_w // 4 + 1
            label_length[i] = label_len

        inputs = {'input_image': images,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  }
        outputs = {'ctc': np.zeros([self.batch_size])}

        return inputs, outputs


if __name__ == '__main__':
    # for i in range(100):
    #     x,y = data_gen()
    a = DATASequence()
    a.__getitem__(6)

