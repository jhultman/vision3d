import glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if __name__ == '__main__':
    setup(
        name='pvrcnn',
        version='0.1',
        description='Implementation of PV-RCNN algorithm',
        author='Jacob Hultman',
        packages=find_packages(),
        package_data={'pvrcnn.ops': ['*/*.so']},
        install_requires=[
            'numpy', 'torch', 'yacs', 'tqdm', 'spconv', 'pointnet2', 'torchsearchsorted'],
        ext_modules=[
            CUDAExtension(
                'pvrcnn._C',
                glob.glob('pvrcnn/ops/rotated_iou/*.cpp') +
                glob.glob('pvrcnn/ops/rotated_iou/*.cu'),
                extra_compile_args={'cxx': ['-g',], 'nvcc': ['-g',],},
            ),
        ],
        cmdclass={'build_ext': BuildExtension},
    )
