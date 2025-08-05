## Requirements and dependencies

```
Python 3.8-3.10
Windows x32 or x64
numpy>=1.19.5
pytorch>=1.8.1
torchvision>=0.9.0
cvxopt>=1.3.0
matplotlib>=3.1.3
```

## Basic usage
To replicate the results of experiments, run the following command in Terminal by navigating the directory to same folder.
```
python run.py --seed 1 --device 0 --module CNN --algorithm CoReFed --dataloader DataLoader_cifar10_dir --N 100 --Diralpha 0.5 --B 50 --C 0.2 --R 1000 --E 1 --lr 0.1 --decay 0.999
```
All parameters are mentioned in `main.py`.


