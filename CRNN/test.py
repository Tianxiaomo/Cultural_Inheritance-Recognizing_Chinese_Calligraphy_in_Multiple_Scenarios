import sys, os
import time

sys.path.append(os.getcwd())

# crnn packages
import torch
from torch.autograd import Variable
from PIL import Image
import CRNN.crnn as crnn
# from pytorch_model import alphabets, utils, dataset
from CRNN import cfg
from CRNN import utils,dataset

# str1 = alphabets.alphabet

import torchvision

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--images_path', type=str, default='1.jpg', help='the path to your images')
opt = parser.parse_args()


# crnn params
# 3p6m_third_ac97p8.pth
crnn_model_path = 'crnn_19_0.5709116458892822_56.58.pth'
# alphabet = str1
# nclass = len(alphabet)+1


# crnn文本信息识别
def crnn_recognition(cropped_image, model):

    # converter = utils.strLabelConverter(alphabet)
    converter = utils.strLabelConverter(cfg.dic_path)

    image = cropped_image.convert('RGB')
    # image = cropped_image
    ## 
    w = int(image.size[0] / (280 * 1.0 / 480))
    transformer = dataset.resizeNormalize((w, 48))
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)
    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('results: {0}'.format(sim_pred))


if __name__ == '__main__':
	# crnn network
    model = crnn.CRNN(cfg.imgH, cfg.nc,cfg.nclass, cfg.nh)

    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from {0}'.format(crnn_model_path))
    # 导入已经训练好的crnn模型
    model.load_state_dict(torch.load(crnn_model_path,torch.device('cpu')))

    print(model)

    started = time.time()
    ## read an image
    image = Image.open(opt.images_path)

    crnn_recognition(image, model)
    finished = time.time()
    print('elapsed time: {0}'.format(finished-started))
