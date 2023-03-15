import numpy as np


def load_calib():
    """
    Temporarily hardcoding the calibration matrix using calib file from 2011_09_26
    """
    calib = open(f"./kitti_utils/calib_cam_to_cam.txt", "r")
    lines = calib.readlines()
    P_rect_line = lines[25]

    Proj_str = P_rect_line.split(":")[1].split(" ")[1:]
    Proj = np.reshape(np.array([float(p) for p in Proj_str]), (3, 4)).astype(np.float32)
    K = Proj[:3, :3]  # camera matrix

    # note: we will take the center crop of the images during augmentation
    # that changes the optical centers, but not focal lengths
    K[0, 2] = (
        K[0, 2] - 13
    )  # from width = 1242 to 1216, with a 13-pixel cut on both sides
    K[1, 2] = (
        K[1, 2] - 11.5
    )  # from width = 375 to 352, with a 11.5-pixel cut on both sides
    return K
