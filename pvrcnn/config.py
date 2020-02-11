from yacs.config import CfgNode as CN

_C = CN()

# Misc
_C.C_IN = 4
_C.NUM_KEYPOINTS = 2048
_C.STRIDES = [1, 2, 4, 8]
_C.SAMPLES_PN = [16, 32]

# Voxelization
_C.MAX_VOXELS = 40000
_C.MAX_OCCUPANCY = 5
_C.VOXEL_SIZE = [0.05, 0.05, 0.1]
_C.GRID_BOUNDS = [0, -40, -3, 64, 40, 1]

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
    [[1, 8, 8], [1, 8, 8]],
    [[4, 8, 16], [4, 8, 16]],
    [[32, 32, 32], [32, 32, 32]],
    [[64, 64, 64], [64, 64, 64]],
    [[64, 96, 128], [64, 96, 128]]
]

# RoiGridPool parameters
_C.GRIDPOOL = CN()
_C.GRIDPOOL.NUM_GRIDPOINTS = 216
_C.GRIDPOOL.RADII_PN = [0.8, 1.6]
_C.GRIDPOOL.MLPS_PN = [[624, 128, 128], [624, 128, 128]]
_C.GRIDPOOL.MLPS_REDUCTION = [_C.GRIDPOOL.NUM_GRIDPOINTS * 256, 256, 256]

cfg = _C
