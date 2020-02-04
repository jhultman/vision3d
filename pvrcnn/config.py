class PvrcnnConfig:

    C_in = 4
    n_keypoints = 2048
    strides = [1, 2, 4, 8]
    max_num_points = 3
    max_voxels = 40000
    voxel_size = [0.05, 0.05, 0.1]
    grid_bounds = [0, -40, -3, 64, 40, 1]
    sample_fpath = '../data/sample.bin'

    # PointSetAbstraction parameters
    radii = [
        [0.4, 0.8],
        [0.4, 0.8],
        [0.8, 1.2],
        [1.2, 2.4],
        [2.4, 4.8]
    ]
    mlps = [
        [[1, 16, 32], [1, 16, 32]],
        [[16, 16, 32], [16, 16, 32]],
        [[16, 32, 64], [16, 32, 64]],
        [[32, 64, 64], [32, 64, 64]],
        [[64, 64, 128], [64, 64, 128]]
    ]
    nsamples = [[16, 32]] * len(radii)
    assert len(radii) == len(mlps) == len(nsamples)
