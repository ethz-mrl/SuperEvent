import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch

from models.util import box_nms
from util.train_utils import list2device

def fix_seed():
    import random
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

def extract_keypoints_and_descriptors(pts, desc_map, detection_threshold, resize_factor=1.0):
    if len(pts) > 0 and not len(pts[0]) == 2:
        # Keypoint map provided
        pts = np.where(pts > detection_threshold)
        pts = np.array(pts).transpose()
    descriptors = desc_map[pts[:, 0], pts[:, 1]]
    kpts = [cv2.KeyPoint(float(resize_factor * pts[i][1]), float(resize_factor * pts[i][0]), 1) for i in range(len(pts))]
    return kpts, descriptors