# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L


class LeNets(chainer.Chain):

    def __init__(self):

        super(LeNets, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(None, 32, ksize=5, stride=1, pad=2)
            self.conv1_2 = L.Convolution2D(None, 32, ksize=5, stride=1, pad=2)
            self.conv2_1 = L.Convolution2D(None, 64, ksize=5, stride=1, pad=2)
            self.conv2_2 = L.Convolution2D(None, 64, ksize=5, stride=1, pad=2)
            self.conv3_1 = L.Convolution2D(None, 128, ksize=5, stride=1, pad=2)
            self.conv3_2 = L.Convolution2D(None, 128, ksize=5, stride=1, pad=2)
            self.fc1 = L.Linear(None, 2)
            self.fc2 = L.Linear(None, 10)

    def __call__(self, x):

        h = self.extract_feature(x)
        h = F.prelu(h)
        h = self.fc2(h)

        return h

    def extract_feature(self, x):

        h = self.conv1_1(x)
        h = F.max_pooling_2d(self.conv1_2(h), 2, stride=2, pad=0)

        h = self.conv2_1(h)
        h = F.max_pooling_2d(self.conv2_2(h), 2, stride=2, pad=0)

        return self.fc1(h)
