# Exploration of Mixup in SSL
The implementation is based on an unofficial PyTorch [implementation](https://github.com/YU1ut/MixMatch-pytorch) of [MixMatch](https://arxiv.org/abs/1905.02249). 

Now only experiments on CIFAR-10 are available.

## Requirements
- Python 3.6+
- PyTorch 1.3.1
- torchvision 0.4.2
- tensorboardX
- progress
- matplotlib
- numpy

## Usage

### Manifold Mixup.
Train the model by 1000 labeled data of CIFAR-10 dataset:

```
source prepare_env.sh
python train.py --gpu <gpu_id> --n-labeled 1000 --out cifar10@1000-0 --layers_mix 0
```

- Arguments for `layers_mix`: 0,1,2,-1,-2. 0,1,2 means mixup on `x`, `conv1`, `block1` respectively. `-1` means mixup selectively on `x` and `conv1`. `-2` means mixup selectively on `x`, `conv1` and `block1`.

### Consistency between classifiers.

```
source prepare_env.sh
python train_gmmmix.py --gpu <gpu_id> --n-labeled 1000 --out cifar10-1000-0-gmm --layers_mix 0
```


### Monitoring training progress
```
tensorboard.sh --port 6006 --logdir cifar10@250
```

## References
```
@article{berthelot2019mixmatch,
  title={MixMatch: A Holistic Approach to Semi-Supervised Learning},
  author={Berthelot, David and Carlini, Nicholas and Goodfellow, Ian and Papernot, Nicolas and Oliver, Avital and Raffel, Colin},
  journal={arXiv preprint arXiv:1905.02249},
  year={2019}
}
```
