# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L

from center_loss import CenterLoss


class LeNets(chainer.Chain):

    def __init__(self, out_dim, alpha_ratio, lambda_ratio=0.5, is_center_loss=True):

        self.out_dim = out_dim
        self.alpha_ratio = alpha_ratio
        self.lambda_ratio = lambda_ratio
        self.is_center_loss = is_center_loss

        super(LeNets, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(None, 32, ksize=5, stride=1, pad=2)
            self.conv1_2 = L.Convolution2D(None, 32, ksize=5, stride=1, pad=2)
            self.conv2_1 = L.Convolution2D(None, 64, ksize=5, stride=1, pad=2)
            self.conv2_2 = L.Convolution2D(None, 64, ksize=5, stride=1, pad=2)
            self.conv3_1 = L.Convolution2D(None, 128, ksize=5, stride=1, pad=2)
            self.conv3_2 = L.Convolution2D(None, 128, ksize=5, stride=1, pad=2)
            self.fc1 = L.Linear(None, 2)
            self.fc2 = L.Linear(None, out_dim)

            if is_center_loss:
                self.center_loss_function = CenterLoss(alpha_ratio, out_dim)

    def extract_feature(self, x):

        h = self.conv1_1(x)
        h = F.max_pooling_2d(self.conv1_2(h), 2, stride=2, pad=0)
        h = self.conv2_1(h)
        h = F.max_pooling_2d(self.conv2_2(h), 2, stride=2, pad=0)
        h = self.fc1(h)

        return h

    def __call__(self, x, t):

        h = self.extract_feature(x)

        if self.is_center_loss:
            center_loss = self.center_loss_function(h, t)
            chainer.report({"centerloss": center_loss}, self)

        h = F.relu(h)
        h = self.fc2(h)

        softmax_loss = F.softmax_cross_entropy(h, t)
        chainer.report({"softmaxloss": softmax_loss}, self)

        if self.is_center_loss:
            loss = softmax_loss + self.lambda_ratio * center_loss
        else:
            loss = softmax_loss

        chainer.report({"loss": loss, "accuracy": F.accuracy(h, t)}, self)
        return loss
