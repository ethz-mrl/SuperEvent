#!/usr/bin/env python
import argparse
from glob import glob
import numpy as np
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("in_dir", help="Input path")
args = parser.parse_args()

# Find all directories that containing the keypoint maps
matches_dir_list = []
for root, dirs, _ in os.walk(args.in_dir):
    matches_dir_list[0:0] = [os.path.join(root, dir) for dir in dirs if dir == "kp_maps"]

for matches_dir in tqdm(matches_dir_list, desc="Progress of all sequences"):
    output_file_name = os.path.join(matches_dir, "../valid_indeces.txt")

    if os.path.exists(output_file_name):
        print(f"{output_file_name} already exists. Skipping.")
    else:
        kp_map_path_list = sorted(glob(os.path.join(matches_dir, "*.npy")))
        valid_indeces = [j for j, kp_map_path in enumerate(tqdm(kp_map_path_list, desc=f"Processing {matches_dir}")) if np.max(np.load(kp_map_path) > 0.)]

        # Save indeces in sequence folder
        print(f"Saving valid indeces to file {output_file_name}.")
        np.savetxt(output_file_name, valid_indeces, fmt='%.0d')