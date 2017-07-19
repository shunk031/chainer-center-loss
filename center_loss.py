# -*- coding: utf-8 -*-

import numpy

from chainer import function
from chainer.utils import type_check


class CenterLossFunction(function.Function):

    """Center loss function."""

    def check_type_forward(self, in_types):
        pass

    def forward(self, inputs):
        pass


def center_loss_function(x, alpha):
    """Center loss function.

    This function computes center loss.
    """
    return CenterLossFunction(alpha)(x)
