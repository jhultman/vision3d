Install dependencies:
- CUDA 9.0+
- torch 1.0
- Conda (recommended)
- Ubuntu 18.04 (recommended)

Install Pointnet2:
```
git clone https://github.com/sshaoshuai/Pointnet2.PyTorch.git
cd Pointnet2.PyTorch && python setup.py install
export PYTHONPATH=$PYTHONPATH:/path/to/Pointnet2.PyTorch/
```

Install spconv:

```
git clone https://github.com/traveller59/spconv.git --recursive
cd spconv && git checkout 7342772
python setup.py bdist_wheel
cd ./dist && pip install *.whl
```
