#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: 
@file: test.py
@time: 2019/1/10 10:08
@desc:
'''
# import pudb; pu.db
from keras.preprocessing import image
import numpy as np
import time

import cfg
from model import CRNN,DenseNet,CRNN_my
from util.util import gpuConfig
from dataset import DATASequence

img_path = 'test/2.png'

with open(cfg.char_path, 'r',encoding='utf-8') as file:
    char_dict = {num: char.strip() for num, char in enumerate(file.readlines())}

gpuConfig(1)

def test(model):

    img = image.load_img(img_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    new = time.time()
    x = model.predict(img)
    print('predict time %f' % (time.time() - new))

    a = x.argmax(2)[0]
    ctc_end = len(char_dict)-1
    text = [char_dict.get(a[i]) for i in range(len(a)) if a[i]!=0 and a[i]!=ctc_end and a[i]!= a[i-1]]
    print(text)

    data = DATASequence(train=False)
    new = time.time()
    model.predict_generator(data,steps=100,verbose=1)
    print('predict time %f' % (time.time() - new))


if __name__ == '__main__':
    for i in dir(cfg):
        print(i, eval('cfg.' + i))
    model = None
    if cfg.model == 'CRNN':
        model = CRNN(cfg.checkpoint).model
    elif cfg.model == 'DesenNet':
        model = DenseNet('checkpoints/weights.011-0.449-0.85.h5').model
    elif cfg.model == 'CRNN_my':
        model = CRNN_my('checkpoints/weights_CRNN_my.040-0.087-0.987.h5').model
    else:
        print('%s is not define!!!' % cfg.model)
    test(model)
