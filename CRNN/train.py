from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from torch_baidu_ctc import CTCLoss

from torchnet import meter as tnt

from CRNN import utils, cfg, dataset
import CRNN.crnn as crnn

from utils import ShowProcess
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def trainBatch(model, criterion, optimizer, train_dataloader):
    data, label = train_dataloader
    batch_size = data.size(0)
    t, l = converter.encode(label)
    if cfg.cuda:
        data = data.cuda()
    preds = model(data)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, t, preds_size, l) / batch_size
    model.zero_grad()
    cost.backward()
    optimizer.step()

    acc = 0
    preds = preds.argmax(2).cpu()
    z = torch.tensor(0)
    for i in range(preds.shape[1]):
        t = [preds[j, i] for j in range(preds.shape[0]) if preds[j, i] != z and preds[j - 1, i] != preds[j, i]]
        if len(t) == l[i]:
            if list(torch.stack(t,0).numpy()) == list(converter.encode(label[i])[0].numpy()):acc+=1
    return cost.cpu(),acc


def valBatch(model, criterion,val_data):
    data, label = val_data
    batch_size = data.size(0)
    t, l = converter.encode(label)
    if cfg.cuda:
        data = data.cuda()
    preds = model(data)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, t, preds_size, l) / batch_size

    acc = 0
    preds = preds.argmax(2).cpu()
    z = torch.tensor(0)
    for i in range(preds.shape[1]):
        t = [preds[j, i] for j in range(preds.shape[0]) if preds[j, i] != z and preds[j - 1, i] != preds[j, i]]
        if len(t) == l[i]:
            if list(torch.stack(t, 0).numpy()) == list(converter.encode(label[i])[0].numpy()): acc += 1
    return cost.cpu(),acc


def train(model,criterion,opt,epochs,steps_per_epoch,validation_steps,train_dataloader,val_dataloader):
    ShowProcess.set(steps_per_epoch,epochs)
    loss_meter = tnt.AverageValueMeter()
    acc_meter = tnt.AverageValueMeter()

    train_iter = iter(train_dataloader)
    val_iter   = iter(val_dataloader)
    for epoch in range(epochs):
        loss_meter.reset()
        acc_meter.reset()
        # train
        for step in range(steps_per_epoch):
            data = train_iter.next()
            for p in model.parameters():
                p.requires_grad = True
            model.train()
            loss,acc = trainBatch(model,criterion,opt,data)
            loss_meter.add(loss.data)
            acc_meter.add(acc)

            ShowProcess.show_process(epoch=epoch,step=step+1,loss_train=loss_meter.value()[0].cpu(),acc_train=acc_meter.value())
        # val
        val_loss = 0
        val_acc = 0
        for step in range(validation_steps):
            for p in model.parameters():
                p.requires_grad = False
            model.eval()
            val_data = val_iter.next()
            ret = valBatch(model, criterion,val_data)
            val_loss += ret[0]
            val_acc += ret[1]
        val_loss = val_loss / validation_steps
        val_acc = val_acc/validation_steps
        print('val_loss:%f ,val_acc:%f' % (val_loss,val_acc))
        # save
        torch.save(model.state_dict(),'{0}/crnn_{1}_{2}_{3}.pth'.format(cfg.checkpoints, epoch,val_loss,val_acc))


if __name__ == '__main__':

    # 0：设置随机种子
    manualSeed = random.randint(1, 10000)  # fix seed
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    cudnn.benchmark = True

    # 1：数据加载
    train_dataset = dataset.genDataset1()
    val_dataset = dataset.genDataset1()
    assert train_dataset
    if not cfg.random_sample:
        sampler = dataset.randomSequentialSampler(train_dataset, cfg.batchSize)
    else:
        sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batchSize,
        shuffle=True, sampler=sampler,num_workers=int(cfg.workers),
        collate_fn=dataset.alignCollate(imgH=cfg.imgH, imgW=cfg.imgW, keep_ratio=cfg.keep_ratio,cuda=cfg.cuda))

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batchSize,
        shuffle=True, sampler=sampler,num_workers=int(cfg.workers),
        collate_fn=dataset.alignCollate(imgH=cfg.imgH, imgW=cfg.imgW, keep_ratio=cfg.keep_ratio,cuda=cfg.cuda))

    converter = utils.strLabelConverter(cfg.dic_path)

    # 2：loss
    criterion = CTCLoss()

    # 3：模型
    crnn = crnn.CRNN(cfg.imgH, cfg.nc,cfg.nclass, cfg.nh)
    if cfg.cuda:
        crnn.cuda()
        criterion = criterion.cuda()

    crnn.apply(weights_init)
    if cfg.loadCheckpoint != None:
        print('loading pretrained model from %s' % cfg.loadCheckpoint)
        crnn.load_state_dict(torch.load(cfg.loadCheckpoint))

    # 4：优化方式
    if cfg.adam:
        optimizer = optim.Adam(crnn.parameters(), lr=cfg.lr,betas=(cfg.beta1, 0.999))
    elif cfg.adadelta:
        optimizer = optim.Adadelta(crnn.parameters(), lr=cfg.lr)
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=cfg.lr)

    # 5：train
    train(crnn,criterion=criterion,opt = optimizer,
          epochs=cfg.epochs,steps_per_epoch=cfg.steps_per_epoch,
          validation_steps=cfg.validation_steps,
          train_dataloader=train_loader,val_dataloader=val_loader)
