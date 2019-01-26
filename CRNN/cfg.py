from utils import strLabelConverter

random_sample = True
keep_ratio = False
adam = True
adadelta = False
saveInterval = 2
valInterval = 800
n_test_disp = 10
displayInterval = 5
experiment = './expr'
crnn = ''
beta1 =0.5
lr = 0.001
niter = 300

# Model
nh = 256                                        #rnn 隐藏单元个数
imgW = 480                                      #图片宽
imgH = 48                                       #图片高
batchSize = 64                                  #batchsize
workers = 24                                    #数据加载进程数
cuda = True                                     #GPU
dic_path = 'char_gb2312.txt'                    #字符字典路径
nc = 3                                          #图片channel
nclass = strLabelConverter(dic_path).lexicon_len        # 字符字典长度
checkpoints = 'checkpoints/'
loadCheckpoint = None

# Train
epochs = 100
steps_per_epoch = 1000
validation_steps = 100