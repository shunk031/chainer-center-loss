# -*- coding: utf-8 -*-

import numpy

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
        batch_size = features.shape[0]

        centers_batch = xp.take(centers, labels, axis=0)

        y = xp.sum(xp.square(features - centers_batch)) / batch_size / 2
        y = xp.asarray(y, dtype=xp.float32)

        return y,

    def backward(self, inputs, gy):
        xp = cuda.get_array_module(*inputs)
        features, labels, centers = inputs
        batch_size = features.shape[0]

        centers_batch = xp.take(centers, labels, axis=0)

        diff = features - centers_batch
        gx0 = diff.astype(xp.float32) / batch_size

        d_cj = []
        for i in range(self.num_classes):
            c_j = centers[i]
            indices = xp.where(labels == i)
            d_cj.append(xp.sum(c_j - features[indices], axis=0) / (1 + indices[0].shape[0]))

        d_cj = self.alpha * xp.asarray(d_cj, dtype=xp.float32)

        return gx0, None, d_cj


class CenterLoss(link.Link):

    def __init__(self, alpha, num_classes):

        super(CenterLoss, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes

        initial_centers = initializers.constant.Zero()
        with self.init_scope():
            self.centers = variable.Parameter(initial_centers, (self.num_classes, 2))

    def __call__(self, x, t, alpha):
        return CenterLossFunction(self.alpha, self.num_classes)(x, t, self.centers)
