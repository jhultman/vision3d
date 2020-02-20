# PV-RCNN
Pytorch implementation of [PV-RCNN](https://arxiv.org/pdf/1912.13192): Point-Voxel Feature Set Abstraction for 3D Object Detection.

![PV-RCNN](images/pvrcnn.png)

## Goals
- Short, readable code.
- Simple, easy-to-use API.
- Reproduce results of paper.

## Status
This repo is still under active development. I will post a pretrained model when codebase stabilizes and results are acceptable.

## Usage
See `inference.py`.

## Installation
See `install.md`.

## Citing
If you use this work in your research, please consider citing:

```
@article{pvrcnnpytorch,
  author={Jacob Hultman},
  title={PV-RCNN PyTorch},
  journal={https://github.com/jhultman/PV-RCNN},
  year={2020}
}
```

and the PV-RCNN paper:

```
@article{shi2019pv,
  author={Shi, Shaoshuai and Guo, Chaoxu and Jiang, Li and Wang, Zhe and Shi, Jianping and Wang, Xiaogang and Li, Hongsheng},
  title={PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection},
  journal={arXiv preprint arXiv:1912.13192},
  year={2019}
}
```

## Contributions
Contributions are welcome.

## License
Please see `license.md`. Note that the code in `pvrcnn/thirdparty` remains under the `detectron2` Apache license.
