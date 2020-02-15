from setuptools import setup

setup(
    name='pvrcnn',
    version='0.1',
    description='Implementation of PV-RCNN algorithm',
    author='Jacob Hultman',
    packages=['pvrcnn'],
    install_requires=['numpy', 'torch', 'yacs', 'tqdm', 'spconv', 'pointnet2', 'torchsearchsorted'],
)
