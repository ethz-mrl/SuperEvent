import cv2
import numpy as np
import os

def check_already_exists(path):
    # Checks for a directory in the root data folder if same directory exists in data split (train, test, val)
    # Returns path where it exists or empty string if it does not exist

    split_names = ["", "train", "val", "test"]

    # Remove trailing backslash
    path = os.path.normpath(path)

    # Split up path
    seq_path, file_name = os.path.split(path)
    dataset_path, seq_name = os.path.split(seq_path)
    data_root_path, dataset_name = os.path.split(dataset_path)

    # Check splits
    for split in split_names:
        check_path = os.path.join(data_root_path, split, dataset_name, seq_name, file_name)
        if os.path.exists(check_path):
            return check_path

    return ""

def get_camera_matrix_and_distortion_coeffs(calib_data):
    # Create calib matrix
    camera_matrix = calib_data[:4]
    distortion_coeffs = calib_data[4:]
    camera_matrix = np.array(
        [
            [camera_matrix[0], 0, camera_matrix[2]],
            [0, camera_matrix[1], camera_matrix[3]],
            [0, 0, 1],
        ]
    )

    return camera_matrix, distortion_coeffs