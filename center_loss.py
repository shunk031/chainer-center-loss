# -*- coding: utf-8 -*-

import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class CenterLossFunction(function.Function):

    """Center loss function."""

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def check_type_forward(self, in_types):
        x_type, t_type = in_types
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            x_type.dtype == numpy.float32,
            t_type.dtype == numpy.int32,
            x_type.shape[0] == t_type.shape[0]
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        features, labels = inputs

    def backward(self, inputs, gy):
        pass


def center_loss_function(x, t, num_classes):
    """Center loss function.

    This function computes center loss.
    """
    return CenterLossFunction(num_classes=num_classes)(x, t)
