"""
Reference: https://github.com/kuixu/kitti_object_vis.

MIT License

Copyright (c) 2019 Kui Xu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from collections import namedtuple


def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects


def read_velo(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan


def read_calib(calib_filename):
    """Return calib as named tuple."""
    calib = CalibObject(calib_filename).astuple
    return calib


def filter_camera_fov(calib, points_):
    """Takes ~3.5 ms in KITTI."""
    keep = points_[:, 0] > 0
    p = points_[keep, :3]
    ones = np.ones_like(p[:, 0:1])
    p = (calib.R0 @ calib.V2C) @ np.c_[p, ones].T
    p = calib.P2 @ np.r_[p, ones.T]
    p = (p / p[2:3])[:2].T
    keep[keep] &= ((p >= 0) & (p <= calib.WH)).all(1)
    return points_[keep]


class Object3d:
    """3d object label"""
    def __init__(self, label_file_line):
        data = label_file_line.split(" ")
        data[1:] = [float(x) for x in data[1:]]
        # extract label, truncation, occlusion
        self.class_name = data[0]  # "Car", "Pedestrian", ...
        self.class_idx = self.class_name_to_idx(self.class_name)
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]   # box height
        self.w = data[9]   # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12] - self.h / 2, data[13])  # location (x,y,z) in camera coord.
        self.dis_to_cam = np.linalg.norm(self.t)
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        self.score = data[15] if data.__len__() == 16 else -1.0
        self.level = self.get_obj_level()

    def class_name_to_idx(self, class_name):
        CLASS_NAME_TO_IDX = {
            "Car": 0,
            "Van": 0,
            "Pedestrian": 1,
            "Person_sitting": 1,
            "Cyclist": 2,
        }
        if class_name not in CLASS_NAME_TO_IDX.keys():
            return -1
        return CLASS_NAME_TO_IDX[class_name]

    def get_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1
        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            self.level_str = "Easy"
            return 1  # Easy
        elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            self.level_str = "Moderate"
            return 2  # Moderate
        elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
            self.level_str = "Hard"
            return 3  # Hard
        else:
            self.level_str = "UnKnown"
            return 4


keys = ['V2C', 'C2V', 'R0', 'P2', 'WH']
Calib = namedtuple('Calib', keys)


class CalibObject:
    """
    3d XYZ in <label>.txt are in rect camera coord.
    Points in <lidar>.bin are in Velodyne coord.
    """

    def __init__(self, calib_filepath):
        calibs = self.read_calib_file(calib_filepath)
        # Rigid transform from Velodyne to reference camera
        self.V2C = calibs["V2C"].reshape(3, 4)
        # Inverse of above
        self.C2V = self.inverse_rigid_trans(self.V2C)
        # Rotation from reference camera to rect camera
        self.R0 = calibs["R0"].reshape(3, 3)
        # Projection matrix from rect camera to image2
        self.P2 = calibs["P2"].reshape(3, 4)
        # Hardcoded image shape (approximate)
        self.WH = np.r_[1224, 370]
        # Serializable representation
        self.astuple = Calib(
            self.V2C, self.C2V, self.R0, self.P2, self.WH)

    def read_calib_file(self, filepath):
        with open(filepath) as f:
            lines = f.readlines()
        obj = lines[2].strip().split(" ")[1:]
        P2 = np.array(obj, dtype=np.float32)
        obj = lines[3].strip().split(" ")[1:]
        P3 = np.array(obj, dtype=np.float32)
        obj = lines[4].strip().split(" ")[1:]
        R0 = np.array(obj, dtype=np.float32)
        obj = lines[5].strip().split(" ")[1:]
        Tr_velo_to_cam = np.array(obj, dtype=np.float32)
        transforms = dict(
            P2=P2.reshape(3, 4),
            P3=P3.reshape(3, 4),
            R0=R0.reshape(3, 3),
            V2C=Tr_velo_to_cam.reshape(3, 4),
        )
        return transforms

    def inverse_rigid_trans(self, Tr):
        """ Inverse a rigid body transform matrix (3x4 as [R|t])
            [R"|-R"t; 0|1]
        """
        inv_Tr = np.zeros_like(Tr)  # 3x4
        inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
        inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
        return inv_Tr
