# PV-RCNN

Pytorch implementation of the algorithm detailed in the
paper PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection
available [here](https://arxiv.org/pdf/1912.13192).

# Installation
Tested in environment:
- CUDA 10.0
- torch 1.0
- Conda
- Ubuntu 18.04
- Python 3.6


1. Installing Pointnet2:
```
git clone https://github.com/sshaoshuai/Pointnet2.PyTorch.git
cd Pointnet2.PyTorch && python setup.py install
export PYTHONPATH=$PYTHONPATH:/path/to/Pointnet2.PyTorch/
```


2. Installing spconv:
```
git clone https://github.com/traveller59/spconv.git --recursive
cd spconv && git checkout 7342772
python setup.py bdist_wheel
cd ./dist && pip install *.whl
```


3. Installing pvrcnn (this package):
```
python setup.py develop
```
