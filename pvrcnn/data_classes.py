import torch


class Pointcloud(torch.Tensor):
    """Convenience wrapper for 3D-pointcloud."""

    def __init__(self, data):
        self.data = data

    @property
    def xyz(self):
        """Return point location."""
        return self.data[..., 0:3]

    @property
    def feature(self):
        """Return rotation around z-axis."""
        return self.data[..., 3:]


class Boxes3D(torch.Tensor):
    """Convenience wrapper for 3D-boxes."""

    def __init__(self, data):
        self.data = data

    @property
    def wlh(self):
        """Return dims in wlh order."""
        return self.data[..., 3:6]

    @property
    def center(self):
        """Return dims in wlh order."""
        return self.data[..., 0:3]

    @property
    def yaw(self):
        """Return rotation around z-axis."""
        return self.data[..., 6:7]
