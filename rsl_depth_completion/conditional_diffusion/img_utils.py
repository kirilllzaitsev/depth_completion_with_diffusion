import os
from glob import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def center_crop(img, crop_size, channels_last=True):
    if channels_last:
        h, w = img.shape[:2]
    else:
        h, w = img.shape[1:3]
    th, tw = crop_size
    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    if channels_last:
        return img[i : i + th, j : j + tw, :]
    return img[:, i : i + th, j : j + tw]
