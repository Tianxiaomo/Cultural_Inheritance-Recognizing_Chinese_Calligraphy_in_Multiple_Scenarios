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
import numpy as np
from keras.utils import Sequence

from data_generator.generator import gen_data
import cfg

class DATA_V_Sequence(Sequence):
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
        maxlabellen = self.img_w//8
        images = np.zeros([self.batch_size, self.img_h, self.img_w, self.img_c])
        labels = np.zeros([self.batch_size, maxlabellen], dtype=np.int32)
        input_length = np.zeros([self.batch_size, 1], dtype=np.int32)
        label_length = np.zeros([self.batch_size, 1])

        for i in range(self.batch_size):
            img, text = gen_data(cfg.data_para)
            img = np.asarray(img)
            _,w,_ = img.shape
            l = (cfg.img_w - w) // 2
            r = cfg.img_w - w - l
            # img = np.pad(img,((0,0),(l,r),(0,0)),'edge')
            img = np.pad(img, ((0, 0), (l, r), (0, 0)), 'constant',constant_values=(0,0))
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
    a = DATA_V_Sequence()
    a.__getitem__(6)

