import torch


class Boxes3D:
    """Convenience wrapper for 3D-boxes."""

    def __init__(self, tensor):
        self._tensor = tensor

    @property
    def wlh(self):
        """Return dims in wlh order."""
        return self.tensor[..., 3:6]

    @property
    def center(self):
        """Return dims in wlh order."""
        return self.tensor[..., 0:3]

    @property
    def yaw(self):
        """Return rotation around z-axis."""
        return self.tensor[..., 6:7]

    @property
    def tensor(self):
        return self._tensor
