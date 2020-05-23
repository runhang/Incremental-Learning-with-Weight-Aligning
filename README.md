# Incremental Learning with Weight Aligning 
pytorch implementation of "Maintaining Discrimination and Fairness in Class Incremental Learning" from https://arxiv.org/abs/1911.07053

# Dataset 
Download Cifar100 dataset from https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

Put meta, train, test into ./data

# Get Started
## Environment
* Python 3.6+
* torch 1.3.1
* torchvision 0.4.2
* CUDA 10.0 & cudnn 7.6.4
* argparse

# Basic Install
```
pip install -r requirements.txt
```

# Usage
```
python main.py
```

# Reference
* https://github.com/sairin1202/BIC

# TODO
- [ ] Add weight clipping