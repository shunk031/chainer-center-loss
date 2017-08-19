# Chainer Center Loss

Implementation of [Center Loss](https://link.springer.com/chapter/10.1007/978-3-319-46478-7_31) in Chainer.

## Requirements

- Python 3.5.1
- Chainer 2.0.0
- CuPy 1.0.0 (if use GPU)
- Matplotlib

## How to train

* with CPU and use center loss

```shell
python train_mnist.py --batchsize 32 --epoch 20 --centerloss
```

* with GPU and use center loss

```shell
python train_mnist.py --batchsize 32 --epoch 20 --gpu 0 --centerloss
```

## Visualize the result

* MNIST, Softmax Loss + Center Loss  
  The white dots denote 10 class centers of deep features.

![Figure 1](https://raw.githubusercontent.com/shunk031/chainer-center-loss/master/img/visualize_deep_feature_with_center_loss.png)

* MNIST, only Softmax Loss

![Figure 2](https://raw.githubusercontent.com/shunk031/chainer-center-loss/master/img/visualize_deep_feature_without_center_loss.png)

## Reference

- [Wen, Yandong, et al. "A discriminative feature learning approach for deep face recognition." European Conference on Computer Vision. Springer International Publishing, 2016.](https://link.springer.com/chapter/10.1007/978-3-319-46478-7_31)
