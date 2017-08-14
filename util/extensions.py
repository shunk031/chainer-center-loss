# -*- coding: utf-8 -*-


import os

from chainer import cuda
from chainer.training import extension
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use("Agg")


class VisualizeDeepFeature(extension.Extension):

    def __init__(self, x, num_classes, is_center_loss):
        self.x = x
        self.num_classes = num_classes
        self.is_center_loss = is_center_loss

    def __call__(self, trainer):

        x = self.x
        num_classes = self.num_classes

        updater = trainer.updater

        visualize_dir = os.path.join(trainer.out, 'visualize')
        if not os.path.isdir(visualize_dir):
            os.mkdir(visualize_dir)
        filename = os.path.join(visualize_dir, '{0:08d}.png'.format(updater.iteration))

        x, labels = updater.converter(x, updater.device)
        model = updater.get_optimizer('main').target
        deep_features = model.extract_feature(x)

        if updater.device >= 0:
            deep_features = cuda.to_cpu(deep_features.data)
            labels = cuda.to_cpu(labels)

        for i in range(num_classes):
            plt.scatter(deep_features[labels == i, 0], deep_features[labels == i, 1], s=2)

        if self.is_center_loss:
            centers = model.center_loss_function.centers
            if updater.device >= 0:
                centers = cuda.to_cpu(centers.data)
            plt.scatter(centers[:, 0], centers[:, 1], c='white')

        plt.legend([str(i) for i in range(num_classes)], loc="upper right")
        plt.savefig(filename)
        plt.clf()
        plt.close()
