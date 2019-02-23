#!/usr/bin/python
# encoding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import collections


class strLabelConverter(object):
    def __init__(self,dic_path):
        with open(dic_path, 'r', encoding='utf-8') as file:
            self.decode_dict = {num: char.strip() for num, char in enumerate(file.readlines())}
            self.encode_dict = {char: num for num, char in self.decode_dict.items()}
            self.lexicon_len = len(self.decode_dict)

    def encode(self, text):
        length = []
        result = []
        for item in text:
            # item = item.decode('utf-8','strict')
            length.append(len(item))
            for char in item:
                try:
                    num = self.encode_dict.get(char)
                    if num is None:
                        num = 0
                    result.append(num)
                except Exception:
                    result.append(0)

        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, a, length, raw=False):
        ctc_end = self.lexicon_len - 1
        text = [self.decode_dict.get(int(a[i])) for i in range(len(a)) if a[i] != 0 and a[i] != ctc_end and a[i] != a[i - 1]]
        return text


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def oneHot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot


def loadData(v, data):
    v.data.resize_(data.size()).copy_(data)
    #print(v.size())


def prettyPrint(v):
    print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
    print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0],
                                              v.mean().data[0]))


def assureRatio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img


import sys, time


class ShowProcess():
    """
    显示处理进度的类
    调用该类相关函数即可实现处理进度的显示
    """
    i = 0  # 当前的处理进度
    max_steps = 0  # 总共需要处理的次数
    max_arrow = 30  # 进度条的长度

    # 初始化函数，需要知道总共的处理次数
    @classmethod
    def set(self, max_steps, epoch_max):
        self.max_steps = max_steps
        self.epoch_max = epoch_max
        self.i = 0
        self.first_time = time.time()

    @classmethod
    def time_f(self, s):
        m = int(s // 60)
        h = int(s // 3600)
        s = int(s % 60)
        if h != 0:
            return '{}:{}:{}'.format(h, m, s)
        elif m != 0:
            return '{}:{}'.format(m, s)
        else:
            return '{}s'.format(s)

    # 显示函数，根据当前的处理进度i显示进度
    @classmethod
    def show_process(self,epoch,step,i=None,**kwargs):
        if i is not None:
            self.i = i
        else:
            self.i += 1

        time_info = self.time_f((time.time() - self.first_time) * (self.max_steps - step))
        self.first_time = time.time()
        num_arrow = int(self.i * self.max_arrow / self.max_steps)
        num_line = self.max_arrow - num_arrow - 1

        epoch_info = '\rEpoch %d/%d  ' % (epoch,self.epoch_max)
        step_info = 'Step %d/%d' % (step,self.max_steps)
        if num_arrow == self.max_arrow:
            pro_info = '[' + '=' * num_arrow + '.' * num_line + ']'
        else:
            pro_info = '[' + '=' * num_arrow + '>' + '.' * num_line + ']'
        loss_info = str(kwargs)
        process_bar = epoch_info + step_info + pro_info + time_info + loss_info

        sys.stdout.write(process_bar)
        sys.stdout.flush()
        if self.i >= self.max_steps:
            self.close()

    @classmethod
    def close(self):
        print('')
        self.i = 0
