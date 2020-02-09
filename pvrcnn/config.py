class PvrcnnConfig:

    def __init__(self):
        self.raw_C_in = 4
        self.n_keypoints = 2048
        self.strides = [1, 2, 4, 8]
        self.max_voxels = 40000
        self.voxel_size = [0.05, 0.05, 0.1]
        self.grid_bounds = [0, -40, -3, 64, 40, 1]
        self.sample_fpath = '../data/sample.bin'

        # PointSetAbstraction parameters
        self.radii = [
            [0.4, 0.8],
            [0.4, 0.8],
            [0.8, 1.2],
            [1.2, 2.4],
            [2.4, 4.8]
        ]
        self.mlps = [
            [[1, 16, 16], [1, 16, 16]],
            [[16, 16, 32], [16, 16, 32]],
            [[16, 32, 32], [16, 32, 32]],
            [[32, 64, 64], [32, 64, 64]],
            [[64, 96, 128], [64, 96, 128]]
        ]
        self.nsamples = [[16, 32]] * len(self.radii)
        assert len(self.radii) == len(self.mlps)

        # RoiGridPool parameters
        self.n_gridpoints = 216
        self.gridpool_samples = [16, 32]
        self.gridpool_radii = [0.8, 1.6]
        self.gridpool_mlps = [[864, 128, 128], [864, 128, 128]]
        self.gridpool_reduction_mlps = [self.n_gridpoints * 256, 256, 256]

        # Voxel feature extractor parameters
        self.max_num_points = 3
        self.vfe_C_in = 4

        # Sparse CNN backbone parameters
        self.cnn_C_in = 4
