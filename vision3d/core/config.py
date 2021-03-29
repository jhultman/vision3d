from yacs.config import CfgNode as CN
import numpy as np

_C = CN()

# Misc
_C.C_IN = 4
_C.NUM_KEYPOINTS = 2048
_C.STRIDES = [1, 2, 4, 8]
_C.SAMPLES_PN = [16, 32]

# Voxelization
_C.MAX_VOXELS = 20000
_C.MAX_OCCUPANCY = 5
_C.VOXEL_SIZE = [0.05, 0.05, 0.1]
_C.GRID_BOUNDS = [0, -40, -3, 70.4, 40, 1]

# Sparse CNN
_C.CNN = 'SpMiddleFHD'

# Classes
_C.ANCHORS = [
    {
        'names': ['Car', 'Van'],
        'wlh': [1.6, 3.9, 1.56],
        'yaw': [0, np.pi / 2],
        'iou_thresh': [0.45, 0.60],
        'score_thresh': 0.3,
        'center_z': -1.0,
    },
    {
        'names': ['Pedestrian', 'Person_sitting'],
        'wlh': [0.6, 0.8, 1.73],
        'yaw': [0, np.pi / 2],
        'iou_thresh': [0.20, 0.35],
        'score_thresh': 0.3,
        'center_z': -0.6,
    },
    {
        'names': ['Cyclist'],
        'wlh': [0.6, 1.76, 1.73],
        'yaw': [0, np.pi / 2],
        'iou_thresh': [0.20, 0.35],
        'score_thresh': 0.3,
        'center_z': -0.6,
    },
]
_C.NUM_PROPOSAL_SAMPLE = -1
_C.ALLOW_LOW_QUALITY_MATCHES = False
_C.NUM_CLASSES = len(_C.ANCHORS)
_C.NUM_YAW = 2
_C.BOX_DOF = 7

# PointSetAbstraction
_C.PSA = CN()
_C.PSA.RADII = [
    [0.4, 0.8],
    [0.4, 0.8],
    [0.8, 1.2],
    [1.2, 2.4],
    [2.4, 4.8]
]
_C.PSA.MLPS = [
    [[1, 8, 16], [1, 8, 16]],
    [[4, 8, 16], [4, 8, 16]],
    [[32, 32, 32], [32, 32, 32]],
    [[64, 64, 64], [64, 64, 64]],
    [[64, 64, 64], [64, 64, 64]]
]

# RoiGridPool parameters
_C.GRIDPOOL = CN()
_C.GRIDPOOL.NUM_GRIDPOINTS = 16
_C.GRIDPOOL.RADII_PN = [0.8, 1.6]
_C.GRIDPOOL.MLPS_PN = [[512, 192, 96], [512, 192, 96]]
_C.GRIDPOOL.MLPS_REDUCTION = [_C.GRIDPOOL.NUM_GRIDPOINTS * 192, 256, 256]

# Proposal
_C.PROPOSAL = CN()
_C.PROPOSAL.C_IN = 128
_C.PROPOSAL.TOPK = 100

# Refinement
_C.REFINEMENT = CN()
_C.REFINEMENT.MLPS = [256, 128]

# Dataset
_C.DATA = CN()
_C.DATA.CACHEDIR = '../data/cache/'
_C.DATA.SPLITDIR = '../data/splitfiles/'
_C.DATA.ROOTDIR = '../data/kitti/training/'

# Train
_C.TRAIN = CN()
_C.TRAIN.LR = 1e-3
_C.TRAIN.LAMBDA = 1.0
_C.TRAIN.EPOCHS = 80
_C.TRAIN.BATCH_SIZE = 6
_C.TRAIN.REFINEMENT_NUM_NEGATIVES = 128

# Data augmentation
_C.AUG = CN()
_C.AUG.GLOBAL_SCALE = [0.95, 1.05]
_C.AUG.GLOBAL_ROTATION = [-np.pi / 4, +np.pi / 4]
_C.AUG.FLIP_HORIZONTAL = True
_C.AUG.DATABASE_SAMPLE = True
_C.AUG.NUM_SAMPLE_OBJECTS = [15, 10, 10]
_C.AUG.MIN_NUM_SAMPLE_PTS = 8

cfg = _C
