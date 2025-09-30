"""
Prepare data for a subset of an Event Camera Dataset sequence
- Undistort images and events
- Create time surfaces
- Create an output directory with undistorted images, undistorted event txt, and time surfaces
"""
import argparse
import cv2
import numpy as np
import os
import sys
from tqdm import tqdm

from util import data_io, helpers

def filter_images(images, time_range, image_timestamps):
    images = [images[i] for i, img_stamp in enumerate(image_timestamps) if time_range[0] <= img_stamp <= time_range[1]]
    image_timestamps_filtered = [img_stamp for img_stamp in image_timestamps if time_range[0] <= img_stamp <= time_range[1]]
    stamps_filtered_out = set(image_timestamps) - set(image_timestamps_filtered)
    if len(stamps_filtered_out) > 0:
        print("Filtered out images with the timestamps", stamps_filtered_out, "since they are outside of the event time range", time_range)
    return images, image_timestamps_filtered

def convert_images(images, calib_data, out_dir, undistort, fisheye_lens_used=False):
    if undistort:
        # Create calib matrix
        camera_matrix, distortion_coeffs = helpers.get_camera_matrix_and_distortion_coeffs(calib_data)

        if fisheye_lens_used:
            image_shape = images[0].shape
            new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K=camera_matrix, D=distortion_coeffs[:4], image_size=image_shape, R=None, balance=1.0)
            valid_image_shape = helpers.calculate_valid_image_shape(image_shape, camera_matrix, distortion_coeffs, new_camera_matrix)

    # Undistort images
    images_dir = os.path.join(out_dir, "frames")
    os.makedirs(images_dir)
    print("Converting frames...")
    for img_idx, img in enumerate(tqdm(images)):
        if undistort:
            if fisheye_lens_used:
                img = cv2.fisheye.undistortImage(img, K=camera_matrix, D=distortion_coeffs[:4], Knew=new_camera_matrix)
                img = img[valid_image_shape[0]:valid_image_shape[1], valid_image_shape[2]:valid_image_shape[3]]  # crop invalid pixels
            else:
                img = cv2.undistort(img, cameraMatrix=camera_matrix, distCoeffs=distortion_coeffs)

        filename = f"{str(img_idx).zfill(8)}.png"
        cv2.imwrite(os.path.join(images_dir, filename), img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", help="Input path")
    parser.add_argument("out_dir", help="Output path")
    parser.add_argument("dataset", help="Name of dataset to be used")
    parser.add_argument("--delta_t", nargs="*", default=[0.001], type=float, help="Time delta of time surfaces")
    parser.add_argument("--undistort", default="")
    args = parser.parse_args()

    undistort = args.undistort != ""
    timestamps_path = os.path.join(args.out_dir, "frame_timestamps.txt")

    # Check if corrected files already exist
    if helpers.check_already_exists(os.path.join(args.out_dir, "frames")):
        print("Undistorted image files in for sequence", os.path.basename(args.out_dir), "already exist and will not be overwritten. Please delete them manually to re-run frame conversion.")
        if not helpers.check_already_exists(timestamps_path):
            print("Warning:", os.path.join(timestamps_path), "does not exist and is not created. Please manually create the file with the frame timestamps or automatically create it together with the frames.")
        sys.exit()

    if not os.path.exists(args.in_dir):
        raise FileNotFoundError(f"Sequence directory does not exist for {args.in_dir}")
    assert not os.path.samefile(args.in_dir, args.out_dir)

    required_data = [data_io.RequiredData.event_time_range, data_io.RequiredData.images, data_io.RequiredData.image_stamps]
    if undistort:
        required_data.append(data_io.RequiredData.calib)

    event_time_range, images, image_timestamps, calib = data_io.load_dataset(args.dataset, args.in_dir, required_data)
    fisheye_lens = args.dataset=="mvsec" or args.dataset=="griffin"

    # Filter events earlier than first time surface delta channel
    images, image_timestamps = filter_images(images, event_time_range + np.min(args.delta_t), image_timestamps)
    # Save timestamps
    np.savetxt(timestamps_path, image_timestamps)

    convert_images(images, calib, args.out_dir, undistort, fisheye_lens_used=fisheye_lens)
