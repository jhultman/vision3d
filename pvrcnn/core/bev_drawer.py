import cv2
import numpy as np

from .geometry import box3d_to_bev_corners


def clipped_percentile(x, p=1):
    """Transform to unit interval robustly."""
    p0, p1 = np.percentile(x, [p, 100 - p])
    x = (np.clip(x, p0, p1) - p0) / (p1 - p0 + 1e-1)
    return x


def make_bev_map(points, pixel_size, bounds):
    """Scatter points to create sparse occupancy image."""
    mask = ((points > bounds[:2]) & (points < bounds[2:])).all(1)
    shape = np.int32(np.ceil((bounds[2:] - bounds[:2]) / pixel_size))[::-1]
    pixels = np.int32(np.floor((points[mask] - bounds[:2]) / pixel_size))
    pixels, counts = np.unique(pixels, return_counts=True, axis=0)
    bev_map = np.zeros(shape, dtype=np.float32)
    bev_map[tuple(pixels[:, ::-1].T)] = counts
    bev_map = clipped_percentile(bev_map)
    return bev_map


class Drawer:
    """Draw BEV occupancy map with boxes. Store in image attribute."""

    def __init__(self,
                 points,
                 boxes=[],
                 labels=[],
                 pixel_size=np.r_[0.1, 0.1],
                 bounds=np.r_[0, -30, 60, 30]):
        self.pixel_size = pixel_size
        self.bounds = bounds
        self.line_kw = dict(thickness=2)
        self.text_kw = dict(
            fontScale=0.6,
            thickness=2,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        )
        self.image = self.build_bev(points)
        self.draw(boxes, labels)

    def build_bev(self, points):
        image = make_bev_map(
            points[:, :2], self.pixel_size, self.bounds)
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

    def get_text_color(self, n):
        color = np.tile(np.r_[255, 0, 0][None], (n, 1))
        color = [list(map(int, c)) for c in color]
        return color

    def get_line_color(self, n):
        color = np.tile(np.r_[0, 255, 0][None], (n, 1))
        color = [list(int(c) for c in ci) for ci in color]
        return color

    def draw_text(self, locs, labels):
        colors = self.get_text_color(locs.shape[0])
        locs = map(tuple, locs.astype(np.int32).tolist())
        for label, loc, color in zip(labels, locs, colors):
            cv2.putText(self.image, label, loc, color=color, **self.text_kw)

    def draw_lines(self, lines):
        colors = self.get_line_color(lines.shape[0])
        lines = map(tuple, lines.astype(np.int32).tolist())
        for line, color in zip(lines, colors):
            cv2.line(self.image, line[0:2], line[2:4], color, **self.line_kw)

    def draw(self, boxes_, labels):
        """No support yet for labels."""
        for boxes in boxes_:
            extent = self.bounds[2:] - self.bounds[:2]
            factor = np.r_[self.image.shape[:2]][::-1] / extent
            corners = (box3d_to_bev_corners(boxes) - self.bounds[:2]) * factor
            line_loc = corners[:, [1, 2, 2, 3, 3, 0]].reshape(-1, 4)
            text_loc = corners[:, 0] + 0.3
            self.draw_lines(line_loc)
            self.draw_text(text_loc, labels=[])
