#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.autograd import Variable

from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import six
import sys
from PIL import Image
import numpy as np
from data_generator import FakeTextDataGenerator
import os


with open('../gen_data/poem_pure.txt','r',encoding='utf-8') as text:
    poem = [i.split('\n')[0] for i in text.readlines()]
    poem_len = len(poem)-1

with open('../gen_data/idiom_pure.txt','r',encoding='utf-8') as text:
    idiom = [i.split('\n')[0] for i in text.readlines()]
    idiom_len = len(idiom)-1

fonts_list = os.listdir('../gen_data/TextRecognitionDataGenerator/fonts')

pics = os.listdir('../gen_data/TextRecognitionDataGenerator/img')
bgs = []
for i in pics:
    bgs.append(Image.open('../gen_data/TextRecognitionDataGenerator/img/'+i))


# 从文字库中随机选择n个字符
def sto_choice_from_info_str(quantity=10):
    if random.random() > (4/quantity):
        text = poem[random.randint(0,poem_len)]
        start = random.randint(0, len(text) - quantity)
        return text[start:start+random.randint(int(quantity*0.8),quantity)]
    else:
        return idiom[random.randint(0,idiom_len)]

class genDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.nSamples = 1000000
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        text = sto_choice_from_info_str(10)

        font = random.sample(fonts_list, 1)[0]
        font = os.path.join('../gen_data/TextRecognitionDataGenerator/fonts', font)

        size = 48
        width = size * 10

        skewing_angle = random.randint(0, 5)

        blur = random.random() / 2

        background_type = 3

        distorsion_type = random.randint(0, 2)
        distorsion_orientation = random.randint(0, 2)

        alignment = random.randint(0, 2)

        text_color = '#000000'

        orientation = 0
        space_width = 1

        bg = random.sample(bgs,1)[0]

        img = FakeTextDataGenerator.generate(text,font,size,skewing_angle,background_type,
                            distorsion_type,distorsion_orientation,width,alignment,text_color,orientation,space_width,bg,blur)

        # img = self.transform(img)

        label = text

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, label)


class lmdbDataset(Dataset):

    def __init__(self, root=None, transform=None, target_transform=None):
        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:

            str = 'num-samples'
            nSamples = int(txn.get(str.encode()))
            self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)

            label_key = 'label-%09d' % index
            label = txn.get(label_key.encode())

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (img, label)


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=32, imgW=256, keep_ratio=False, min_ratio=1,cuda=None):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.cuda = cuda

    def __call__(self, batch):
        images, labels = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)
        return images, labels
