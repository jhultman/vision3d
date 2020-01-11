Install dependencies:
- CUDA 9.0+
- torch 1.0

Install Pointnet2:
`git clone ...; python setup.py install`

Add Pointnet2 to PYTHONPATH:
`export PYTHONPATH=$PYTHONPATH:/path/to/Pointnet2.PyTorch/`

Install spconv:
`git clone ... --recursive; cd spconv; git checkout 7...`
`python setup.py bdist_wheel`
