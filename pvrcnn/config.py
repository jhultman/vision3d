class PvrcnnConfig:

    def __init__(self):
        self.C_in = 4
        self.n_keypoints = 2048
        self.strides = [1, 2, 4, 8]
        self.max_num_points = 3
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
            [[1, 16, 32], [1, 16, 32]],
            [[16, 16, 32], [16, 16, 32]],
            [[16, 32, 64], [16, 32, 64]],
            [[32, 64, 64], [32, 64, 64]],
            [[64, 64, 128], [64, 64, 128]]
        ]
        self.nsamples = [[16, 32]] * len(self.radii)
        assert len(self.radii) == len(self.mlps)
