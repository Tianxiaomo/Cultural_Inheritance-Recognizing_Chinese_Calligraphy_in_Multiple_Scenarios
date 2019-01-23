## 文化传承—汉字书法多场景识别(Cultural Inheritance – Recognizing Chinese Calligraphy in Multiple Scenarios)-Baseline

[Competition URL](https://www.datafountain.cn/competitions/334/details)

使用[East](http://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1704.03155.pdf)定位，[crnn](https://arxiv.org/abs/1507.05717)识别文字

## East

参考[AdvancedEASR](https://github.com/huoyijie/AdvancedEAST)

python preprocess.py, resize image to 256256,384384,512512,640640,736*736, and train respectively could speed up training process.

## crnn

cnn+blstm+crc