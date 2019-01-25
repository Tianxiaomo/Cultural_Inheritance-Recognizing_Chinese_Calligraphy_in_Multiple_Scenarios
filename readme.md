## 文化传承—汉字书法多场景识别(Cultural Inheritance – Recognizing Chinese Calligraphy in Multiple Scenarios)-Baseline

[Competition URL](https://www.datafountain.cn/competitions/334/details)

使用[East](http://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1704.03155.pdf)定位，[crnn](https://arxiv.org/abs/1507.05717)识别文字

## East

参考[AdvancedEASR](https://github.com/huoyijie/AdvancedEAST)

## CRNN

cnn+blstm+crc

使用pytorch，需要安装 ctc组建，有[torch-baidu-ctc](https://pypi.org/project/torch-baidu-ctc/)或[wrap-ctc](https://github.com/SeanNaren/warp-ctc)

### torch-baidu-ctc

可以直接使用pip安装
```angular2html
pip install torch-baidu-ctc
```
但是windows不可以用，作者说的pytorch的原因。

### wrap-ctc

```angular2html
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc
mkdir build; cd build
cmake ..
make
```
也是不支持win，在issue看到有大佬改了[win版本](https://github.com/amberblade/warp-ctc/)的，试了也不行，大佬可以尝试一下