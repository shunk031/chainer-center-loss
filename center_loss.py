# -*- coding: utf-8 -*-

import numpy

import chainer.functions as F
from chainer import cuda
from chainer import function
from chainer import initializers
from chainer import link
from chainer import variable
from chainer.utils import type_check


class CenterLossFunction(function.Function):
    """Center loss function."""

    def __init__(self, alpha, num_classes):
        self.alpha = alpha
        self.num_classes = num_classes

    def check_type_forward(self, in_types):
        x_type, t_type, c_type = in_types
        type_check.expect(in_types.size() == 3)
        type_check.expect(
            x_type.dtype == numpy.float32,
            t_type.dtype == numpy.int32,
            c_type.dtype == numpy.float32,
            x_type.shape[0] == t_type.shape[0],
            c_type.shape[0] == self.num_classes,
            c_type.shape[1] == 2
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        features, labels, centers = inputs

        centers_batch = xp.take(centers, labels, axis=0)
        y = xp.sum((features - centers_batch) ** 2) / 2
        y = xp.asarray(y, dtype=xp.float32)

        return y,

    def backward(self, inputs, gy):
        xp = cuda.get_array_module(*inputs)
        features, labels, centers = inputs

        centers_batch = xp.take(centers, labels, axis=0)

        diff = features - centers_batch
        gx0 = diff.astype(xp.float32)

        cj = []
        for i in range(self.num_classes):
            c_j = centers[i]
            indices = xp.where(labels == i)
            _cj = xp.sum(c_j - features[indices], axis=0) / (1 + indices[0].shape[0])
            cj.append(_cj)

        cj = self.alpha * xp.asarray(cj, dtype=xp.float32)

        return gx0, None, cj


class CenterLoss(link.Link):

    def __init__(self, alpha, num_classes):

        super(CenterLoss, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes

        with self.init_scope():
            self.centers = variable.Parameter(0, (self.num_classes, 2))

    def __call__(self, x, t, alpha, num_classes):
        return CenterLossFunction(self.alpha, self.num_classes)(x, t, self.centers)
