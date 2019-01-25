import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    # Inputs hidden units Out
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    #                   32    3   37     256
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        inC = [nc,64, 128, 256, 256, 512, 512]
        outC = [64, 128, 256, 256, 512, 512, 512]

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(inC[0],outC[0],ks[0],ss[0],ps[0])
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(inC[1],outC[1],ks[1],ss[1],ps[1])
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(inC[2],outC[2],ks[2],ss[2],ps[2])
        self.bn3   = nn.BatchNorm2d(outC[2])

        self.conv4 = nn.Conv2d(inC[3],outC[3],ks[3],ss[3],ps[3])
        self.pool4 = nn.MaxPool2d((2, 2), (2, 1),(0,1))

        self.conv5 = nn.Conv2d(inC[4],outC[4],ks[4],ss[4],ps[4])
        self.bn5   = nn.BatchNorm2d(outC[4])

        self.conv6 = nn.Conv2d(inC[5],outC[5],ks[5],ss[5],ps[5])
        self.pool6 = nn.MaxPool2d((3, 3), (3, 1), (0, 1))

        self.conv7 = nn.Conv2d(inC[6],outC[6],ks[6],ss[6],ps[6])
        self.bn7   = nn.BatchNorm2d(outC[6])

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv feature
        c1 = self.conv1(input)  #480 48
        c1 = self.relu(c1)
        c1 = self.pool1(c1)     #240 24

        c2 = self.conv2(c1)
        c2 = self.relu(c2)
        c2 = self.pool2(c2)     # 120 12

        c3 = self.conv3(c2)
        c3 = self.relu(c3)
        c3 = self.bn3(c3)       #120 12

        c4 = self.conv4(c3)
        c4 = self.relu(c4)
        c4 = self.pool4(c4)     #121 6

        c5 = self.conv5(c4)
        c5 = self.relu(c5)
        c5 = self.bn5(c5)       #121 6

        c6 = self.conv6(c5)
        c6 = self.relu(c6)
        c6 = self.pool6(c6)     #122 3

        c7 = self.conv7(c6)
        c7 = self.relu(c7)
        c7 = self.bn7(c7)       #120 2

        b, c, h, w = c7.size()
        assert h == 1, "the height of conv must be 1"
        conv = c7.squeeze(2) # b *512 * width
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        # rnn features
        output = self.rnn(conv)
        #print(output.size(0))
        # print(output.size())# width*batch_size*nclass
        return output
