#!/usr/bin/env python

import argparse
import cv2
from glob import glob
import numpy as np
import os
import random
import sys
from tqdm import tqdm

import torch

from data_preparation.SuperGluePretrainedNetwork.models.matching import Matching
from util import helpers

def matches2keypoint_maps(matches, img_shape):
    matches = np.array(matches, dtype=int)

    # Create map of zeros with the feature id in its pixel location
    kp_map0 = np.zeros(img_shape)
    kp_map0[matches[0].transpose()[0], matches[0].transpose()[1]] = np.arange(len(matches[0])) + 1

    kp_map1 = np.zeros(img_shape)
    kp_map1[matches[1].transpose()[0], matches[1].transpose()[1]] = np.arange(len(matches[0])) + 1

    return [kp_map0, kp_map1]

if __name__ == "__main__": 
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", help="Input path")
    parser.add_argument("--confidence_threshold", default=0.2, help="Matches below confidence threshold are filtered out.")
    parser.add_argument("--max_step_size_in_images", default=-1, help="Maximal number of images skipped between match predictions, randomly chosen between 1 and this value. -1 for default for this sequence.")
    parser.add_argument("--demo", default=False, action=argparse.BooleanOptionalAction, help="Generate data to show advantages of event camera: Few matches between consecutive frames. Requires config to find test sequences.")
    args = parser.parse_args()

    image_path = os.path.join(args.in_dir, "frames")
    if args.demo:
        matches_path = os.path.join(args.in_dir, "demo_matches")
        print("Generating demo data.")
    else:
        matches_path = os.path.join(args.in_dir, "sg_matches")

    existing_path = helpers.check_already_exists(matches_path)
    if existing_path:
        print("Directory", existing_path, "already exists and will not be overwritten. Please delete it manually to filter and convert new matches.")
        sys.exit()
    else:
        os.makedirs(matches_path)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    image_paths = sorted(glob(os.path.join(image_path, "*.png")))
    image_list = [cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype('float32') for path in image_paths]
    image_shape = image_list[0].shape[:2]
    print("Image shape:", image_shape)

    # Convert to tensor
    image_tensor_list = [torch.from_numpy(img/255.).float()[None, None, :, :].to(device) for img in image_list]

    # Create matching function
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        },
        'superglue': {
            'weights': 'indoor',
            'sinkhorn_iterations': 20,
            'match_threshold': args.confidence_threshold,
        }
    }
    
    # Check if outdoor sequence
    sequence_name = os.path.basename(os.path.normpath(args.in_dir))
    dataset_name = os.path.basename(os.path.dirname(os.path.normpath(args.in_dir)))
    outdoor_sequence = "outdoor" in sequence_name or \
                       ("griffin" in dataset_name and not "Testbed" in sequence_name) or \
                       "ddd20" in dataset_name
    if outdoor_sequence:
        config["superglue"]["weights"] = "outdoor"
    
    if args.max_step_size_in_images < 1:
        if "fpv" in dataset_name and "_45_" in sequence_name or \
           "griffin" in dataset_name and "Hills" in sequence_name:
            # Little amount of features in these sequences
            args.max_step_size_in_images = 1
        elif "ddd20" in dataset_name:
            # Few features also in ddd20
            args.max_step_size_in_images = 10
        else:
            args.max_step_size_in_images = 20

    Matcher = Matching(config).eval().to(device)

    # Settings
    min_median_pixel_distance = 1.0
    min_amount_of_matches_required = np.min(image_shape) / 2
    max_amount_of_matches_required = np.inf
    if args.demo:
        # Only use consecutive images to assure visual overlap
        args.max_step_size_in_images = 1
        min_amount_of_matches_required = 0
        if "mvsec" in dataset_name:
            max_amount_of_matches_required = np.min(image_shape) / 1.5
        elif "fpv" in dataset_name:
            max_amount_of_matches_required = np.min(image_shape) / 10
        else:
            max_amount_of_matches_required = np.min(image_shape) / 5

    # First, remove all static images
    filtered_image_tensor_list = []
    filtered_image_paths = []
    print("Filtering out images that are static or do not have enough features.")
    for i, src_image in enumerate(tqdm(image_tensor_list[:-1])):
        res = Matcher({"image0": src_image, "image1": image_tensor_list[i+1]})
        valid = res["matches0"][0] > -1

        # Calculate median of distance of features
        if sum(valid) > min_amount_of_matches_required:
            pixel_distance = torch.linalg.vector_norm(res["keypoints0"][0][valid] - res["keypoints1"][0][res["matches0"][0]][valid], dim=1)  # returns 0. for no matches
            if torch.median(pixel_distance) > min_median_pixel_distance:
                filtered_image_tensor_list.append(image_tensor_list[i+1])
                filtered_image_paths.append(image_paths[i+1])

    print(f"Filtered out {100 * (1 - len(filtered_image_tensor_list) / len(image_tensor_list))}% of images since they were predicted as static or not sufficiently matchable.")

    # Predict and match
    for i, src_image in enumerate(tqdm(filtered_image_tensor_list[:-1])):
        j = i + random.randint(1, args.max_step_size_in_images)
        while j < len(filtered_image_tensor_list):
            # Match images
            res = Matcher({"image0": src_image, "image1": filtered_image_tensor_list[j]})
            valid = res["matches0"][0] > -1

            if sum(valid) > min_amount_of_matches_required and sum(valid) < max_amount_of_matches_required:
                # Save matches of pair
                matches = {k: v[0].cpu().numpy() for k, v in res.items()}
                matches["image_id0"] = os.path.splitext(os.path.basename(filtered_image_paths[i]))[0]
                matches["image_id1"] = os.path.splitext(os.path.basename(filtered_image_paths[j]))[0]
                if args.demo and int(matches["image_id1"]) - int(matches["image_id0"]) > args.max_step_size_in_images:
                    # too many images skipped, might contain many false matches
                    break
                np.savez_compressed(os.path.join(matches_path, matches["image_id0"] + "_" + matches["image_id1"] + ".npz"), **matches)

                j += random.randint(1, args.max_step_size_in_images)

                if args.demo:
                    break
            else:
                # Skip to next image
                break