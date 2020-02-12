import torch


class Boxes3D:
    """Convenience wrapper for 3D-boxes."""

    def __init__(self, tensor):
        self._tensor = tensor

    @property
    def center(self):
        """Return dims in wlh order."""
        return self.tensor[..., 0:3]

    @property
    def wlh(self):
        """Return dims in wlh order."""
        return self.tensor[..., 3:6]

    @property
    def yaw(self, squeeze=True):
        """Return rotation around z-axis."""
        if squeeze:
            return self.tensor[..., 6:7].squeeze(-1)
        return self.tensor[..., 6:7]

    @property
    def score(self):
        """Return rotation around z-axis."""
        return self.tensor[..., 7:8]

    @property
    def tensor(self):
        return self._tensor
