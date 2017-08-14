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

## Reference

- [Wen, Yandong, et al. "A discriminative feature learning approach for deep face recognition." European Conference on Computer Vision. Springer International Publishing, 2016.](https://link.springer.com/chapter/10.1007/978-3-319-46478-7_31)
