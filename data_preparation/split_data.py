#!/usr/bin/env python

import argparse
from glob import glob
import os
import yaml

def move_tree(sourceRoot, destRoot):
    os.makedirs(destRoot, exist_ok=True)
    for path, _, files in os.walk(sourceRoot):
        relPath = os.path.relpath(path, sourceRoot)
        destPath = os.path.join(destRoot, relPath)
        if not os.path.exists(destPath):
            os.makedirs(destPath)
        for file in files:
            destFile = os.path.join(destPath, file)
            srcFile = os.path.join(path, file)
            os.replace(srcFile, destFile)
    for path, _, _ in os.walk(sourceRoot, False):
        os.rmdir(path)  # must be empty, otherwise something went wrong moving

def move_sequences(src_parent_path, target_parent_path, config_sequences):
    for dataset in config_sequences:
        dataset_name = next(iter(dataset))

        for sequence_name in dataset[dataset_name]:
            # Make sure this sequence does not exist in train folder
            assert not os.path.exists(os.path.join(src_parent_path, "train", dataset_name, sequence_name)), \
                f"Sequence {sequence_name} of dataset {dataset_name} already exists in 'train' folder! " \
                 "Please delete/ move it or adjust the val_sequences/ test_sequences in the config file to proceed."

            # Check if src path exists (new data was generated)
            src_path = os.path.join(src_parent_path, dataset_name, sequence_name)
            if not os.path.exists(src_path):
                continue

            # Move sequence and merge with existing folder
            target_path = os.path.join(target_parent_path, dataset_name, sequence_name)
            os.makedirs(target_path, exist_ok=True)
            move_tree(src_path, target_path)
            print("Moved", src_path, "to", target_path)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", help="Input path")
    parser.add_argument("config_path", help="Path to config file")
    args = parser.parse_args()

    # Load config
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
        print("Loaded config from", args.config_path)

    # Remove all empty source directories
    dataset_paths = glob(os.path.join(args.in_dir, "*"))
    for dataset_src_path in dataset_paths:
        sequence_paths = glob(os.path.join(dataset_src_path, "*"))
        for sequence_src_path in sequence_paths:
            if os.path.exists(sequence_src_path):
                try:
                    os.removedirs(sequence_src_path)
                except OSError:
                    pass

    train_data_path = os.path.join(os.path.expanduser(args.in_dir), "train")
    val_data_path = os.path.join(os.path.expanduser(args.in_dir), "val")
    test_data_path = os.path.join(os.path.expanduser(args.in_dir), "test")
    os.makedirs(train_data_path, exist_ok=True)
    os.makedirs(val_data_path, exist_ok=True)
    os.makedirs(test_data_path, exist_ok=True)

    # Move val and test sequences
    move_sequences(args.in_dir, val_data_path, config["val_sequences"])
    move_sequences(args.in_dir, test_data_path, config["test_sequences"])

    # Then, move rest into train
    for dataset_src_path in dataset_paths:
        dataset_name = os.path.basename(dataset_src_path)
        if dataset_name not in ["train", "val", "test"] and os.path.exists(dataset_src_path):
            dataset_target_path = os.path.join(train_data_path, dataset_name)
            move_tree(dataset_src_path, dataset_target_path)
            print("Moved", dataset_src_path, "to", dataset_target_path)