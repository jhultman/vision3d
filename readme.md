# Vision 3D
A clean, easy-to-use PyTorch library for lidar perception. Currently supports SECOND detector.

## Project goals
- Emphasis on simple codebase (no 1,000 LOC functions).
- General 3D detection library (easy to extend to new models and datasets).
- Hope to reproduce state-of-the-art results.

## Status and plans
- At this time I do not have capacity to develop this project. Community support is welcomed.
- I hope this project can serve as useful starting point for lidar perception research.
- Implementation of [PV-RCNN](https://arxiv.org/pdf/1912.13192) is work-in-progress.
- These forks ([one](https://github.com/muzi2045/PV-RCNN), [two](https://github.com/eraofelix/PV-RCNN)) have shown some promise in training on other datasets (NuScenes, and proprietary lidar data).

## Usage
See [inference.py](vision3d/inference.py) and [train.py](vision3d/train.py). To train, need to first start a visdom server using command `visdom` to enable train loss monitoring. (Requires visdom python package to be installed).

## Installation
See [install.md](install.md).

## Preliminary results
![Sample result](/images/sample-result.png)

## Citing
If you find this work helpful in your research, please consider starring this repo and citing:

```
@article{pvrcnnpytorch,
  author={Jacob Hultman},
  title={vision3d},
  journal={https://github.com/jhultman/vision3d},
  year={2020}
}
```

## Contributions
Contributions are welcome. Please post an issue if you find any bugs.

## Acknowledgements and licensing
Please see [license.md](license.md). Note that the code in `vision3d/ops` is largely from [detectron2](https://github.com/facebookresearch/detectron2) and hence is subject to the Apache [license](vision3d/ops/LICENSE).
