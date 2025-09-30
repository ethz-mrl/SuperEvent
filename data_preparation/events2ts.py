#!/usr/bin/env python

import argparse
import cv2
import numpy as np
import os
import sys
from time import time
import torch
from tqdm import tqdm

from data_preparation.util import data_io, helpers
from ts_generation.generate_ts import TsGenerator

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", help="Input directory of sequence where events file is located")
    parser.add_argument("out_dir", help="Output directory of time surfaces")
    parser.add_argument("dataset", help="Name of dataset to be used")
    parser.add_argument("--timestamp_path", default="", help="Path to timestamp file")
    parser.add_argument("--delta_t", nargs="*", default=[], type=float, help="Time delta of time surfaces")
    parser.add_argument("--shape", nargs=2, default=[], type=int, help="Image shape")
    parser.add_argument("--undistort", default="")
    parser.add_argument("--frequency", default=0, type=int, help="Frequency of time surfaces")
    args = parser.parse_args()

    undistort = args.undistort != ""

    # Check if output dir already exists
    existing_path = helpers.check_already_exists(args.out_dir)
    if existing_path:
        print("Directory", existing_path, " already exists and will not be overwritten. Please delete it manually to re-create time surfaces.")
        sys.exit()

    # Read files
    required_data = [data_io.RequiredData.events]
    if args.timestamp_path == "":
        required_data.append(data_io.RequiredData.image_stamps)
    if undistort:
        required_data.append(data_io.RequiredData.calib)
    events, _, image_timestamps, calib = data_io.load_dataset(args.dataset, args.in_dir, required_data)

    # Load timestamp from different path if specified
    if args.timestamp_path:
        existing_timestamp_path = helpers.check_already_exists(args.timestamp_path)
        print("Loading timestamps for time surfaces from", existing_timestamp_path)
        image_timestamps = np.genfromtxt(existing_timestamp_path)

        # Shift time to start with first event (could not be done by data loading script)
        start_time = events[0, 0]
        events[:, 0] = events[:, 0] - start_time
        image_timestamps = image_timestamps - start_time

    if args.frequency:
        print(f"Generating approx. {args.frequency} time surfaces per second.")
        additional_ts_timestamps = []
        for i, timestamp in enumerate(image_timestamps):
            if i == 0:
                continue

            num_additional_ts = round((timestamp - image_timestamps[i-1]) * args.frequency)
            time_delta = (timestamp - image_timestamps[i-1]) / num_additional_ts
            ts_timestamps = [image_timestamps[i-1] + j * time_delta for j in range(1, num_additional_ts)]

            # Assign to empty slice at end of list
            l = len(additional_ts_timestamps)
            additional_ts_timestamps[l:l] = ts_timestamps

    # Convert calib
    if undistort:
        camera_matrix, distortion_coeffs = helpers.get_camera_matrix_and_distortion_coeffs(calib)

    # Cast events to correct data types
    events_t = torch.tensor(events[:, 0].astype(np.float32))
    events_x = torch.tensor(events[:, 2].astype(np.int16))  # we use (row, column) instead of (x, y) coordinates
    events_y = torch.tensor(events[:, 1].astype(np.int16))
    events_p = torch.tensor(events[:, 3].astype(np.int8))
    assert torch.all(events_p >= 0), "Please convert polarity to be [0, 1]."

    # Try to infer shape
    if len(args.shape) == 0:
        args.shape = [torch.max(events_x).item() + 1, torch.max(events_y).item() + 1]
        print("Inferred shape as", args.shape)

    settings = {"shape": args.shape, "delta_t": args.delta_t}
    if undistort and (args.dataset == "mvsec" or args.dataset == "griffin"):
        settings["fisheye_lens"] = True
        settings["new_camera_matrix"] = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K=camera_matrix, D=distortion_coeffs[:4], image_size=args.shape, R=None, balance=1.0)
        settings["crop_to_idxs"] = helpers.calculate_valid_image_shape(args.shape, camera_matrix, distortion_coeffs, settings["new_camera_matrix"])

    if undistort:
        settings["undistort"] = True
        ts_gen = TsGenerator(camera_matrix=camera_matrix, distortion_coeffs=distortion_coeffs, settings=settings)
    else:
        settings["undistort"] = False
        ts_gen = TsGenerator(settings=settings)

    print("Generating time surfaces.")
    start_time = time()

    os.makedirs(args.out_dir)
    print("Saving time surfaces in", args.out_dir)

    image_timestamps_iter = iter(image_timestamps)
    timestamp = next(image_timestamps_iter)
    ts_idx = 0

    if args.frequency:
        additional_timestamps_iter = iter(additional_ts_timestamps)
        additional_timestamp = next(additional_timestamps_iter)
    add_ts_idx = 0

    for i in tqdm(range(len(events_t))):
        if timestamp < events_t[i]:            
            # Create current time surface
            ts = ts_gen.get_ts().numpy()

            # Directly save ts to not run out of RAM
            output_ts_path = os.path.join(args.out_dir, f"{str(ts_idx).zfill(8)}.npy")
            data_io.save_ts_sparse(output_ts_path, ts)
            ts_idx += 1
            add_ts_idx = 0

            timestamp = next(image_timestamps_iter, -1.)
            if timestamp < 0:
                break

        if args.frequency and additional_timestamp < events_t[i]:
            # Create current time surface
            ts = ts_gen.get_ts()

            # Directly save ts to not run out of RAM
            output_ts_path = os.path.join(args.out_dir, f"{str(ts_idx).zfill(8)}_{str(add_ts_idx).zfill(4)}.npy")
            data_io.save_ts_sparse(output_ts_path, ts)
            add_ts_idx += 1

            additional_timestamp = next(additional_timestamps_iter, np.inf)

        # Feed event
        ts_gen.update(events_t[i], events_x[i], events_y[i], events_p[i])

    if timestamp > 0.:
        output_ts_path = os.path.join(args.out_dir, f"{str(ts_idx).zfill(8)}.npy")
        data_io.save_ts_sparse(output_ts_path, ts)
        ts_idx += 1

    time_elapsed = time_elapsed = time() - start_time
    assert ts_idx == len(image_timestamps), str(ts_idx) + " != " + str(len(image_timestamps))
    print("Converted {0} events to {1} time surfaces in {2} seconds.".format(len(events), ts_idx + 1, time_elapsed))
