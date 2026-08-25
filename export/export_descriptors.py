# Script to generate 4 descriptors that can be used to train DBoW2 for okvis2 (loop closure)

import argparse
from glob import glob
import h5py
import hdf5plugin
import math
import numpy as np
import os
from pandas import read_csv
import time
import torch
from tqdm import tqdm
import yaml

from data_preparation.util import helpers
from models.super_event import SuperEvent, SuperEventFullRes
from models.util import fast_nms, interpolate_desc_grid
from ts_generation.ts_generator import TsGenerator, TsGeneratorType
from util.eval_utils import fix_seed

# Fix seed for reproducibility
fix_seed()

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("dataset_path", help="Root directory of dataset")
parser.add_argument("--dataset_name", default="", help="Dataset used for evaluation. Supported datasets are Event Camera Dataset ('ecd') and Event-aided Direct Sparse Odometry ('eds').")
parser.add_argument("--sequence_names", nargs="*", default=[], help="Names of evaluation sequences")
parser.add_argument("--config", default="config/super_event.yaml", help="Parameter configuration.")
parser.add_argument("--out_dir", default="results", help="Directory where results are saved")
parser.add_argument("--model", default="", help="Model weights to be evaluated. If not specified, the most recent weights in saved_models/ are used.")
parser.add_argument("--model_delta_t", nargs="*", default=[0.001, 0.003, 0.01, 0.03, 0.1], type=float, help="Time delta of time surfaces")
parser.add_argument("--model_events_per_px", nargs="*", default=[0.03, 0.1, 0.3, 1.0], type=float, help="Number of events per channel in time surfaces")
parser.add_argument("--mcts_type", default="ne", help="Constant number of events or constant time window duration (dt) per channel.")
parser.add_argument("--max_eval_delta_t", nargs="*", default=2.0, type=float, help="List of durations between testing the repeatability")
parser.add_argument("--auc_thresholds",  nargs="*", default=[5.0, 10.0, 20.0], help="Thresholds for AUC evaluation.")
parser.add_argument('--visualize', default=False, action=argparse.BooleanOptionalAction, help="Visualize matches, only for debugging")
parser.add_argument("--kpts_dir", default="", help="Directory where keypoints are loaded from. If not specified (default), SuperEvent will predict the keypoints.")
parser.add_argument("--num_kpts_per_step", default=-1, help="Number of keypoints tracked in parallel. Only used when 'kpts_dir' is specified. -1 to infer (default).")
args = parser.parse_args()

if not args.dataset_name:
    args.dataset_name = os.path.basename(os.path.normpath(args.dataset_path))
    print(f"Evaluating on dataset {args.dataset_name}.")

supported_datasets = ["ecd", "eds"]
assert args.dataset_name in supported_datasets, f"Datset {args.dataset_name} not supported. Please use one of {supported_datasets}."

if not args.sequence_names:  # Use default sequences
    if args.dataset_name == "ecd":
        args.sequence_names = ["boxes_6dof", "boxes_rotation", "poster_6dof", "poster_rotation", "shapes_6dof", "shapes_rotation"]
    elif args.dataset_name == "eds":
        args.sequence_names = ["peanuts_light", "rocket_earth_light", "ziggy_and_fuzz", "all_characters"]

if args.dataset_name == "ecd":
    ts_shape = [180, 240]
    pose_file_name = "groundtruth.txt"
    events_file_name = "events.txt"
    image_timestampes_file_name = "images.txt"
    ransac_threshold = 1.0
elif args.dataset_name == "eds":
    ts_shape = [480, 640]
    pose_file_name = "stamped_groundtruth.txt"
    events_file_name = "events.h5"
    image_timestampes_file_name = "images_timestamps.txt"
    ransac_threshold = 3.0

if args.kpts_dir:
    experiment_name = "pose_estimation_" + args.dataset_name + "_" + os.path.basename(os.path.normpath(args.kpts_dir)) + "_at_time_" + time.strftime("%Y%m%d-%H%M%S")
else:
    experiment_name = "pose_estimation_" + args.dataset_name + "_" + os.path.splitext(os.path.basename(args.model))[0] + "_at_time_" + time.strftime("%Y%m%d-%H%M%S")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        print("Loaded config from", args.config)
    if "backbone" in config:
        # Load backbone config and add
        backbone_config_path = os.path.join(os.path.dirname(args.config), "backbones", config["backbone"] + ".yaml")
        if os.path.exists(backbone_config_path):
            with open(backbone_config_path, "r") as f:
                backbone_config = yaml.safe_load(f)
                print("Loaded backbone config from", backbone_config_path)
            config = config | backbone_config
            config["backbone_config"]["input_channels"] = config["input_channels"]
        else:
            print("No additional config file found for backbone", config["backbone"])

    # Load model
    if args.model == "":
        # Use most recent model in saved_models
        list_of_files = glob("saved_models/*.pth")
        args.model = max(list_of_files, key=os.path.getctime)
    else:
        if config["pixel_wise_predictions"]:
            model = SuperEventFullRes(config)
        else:
            model = SuperEvent(config)
    model.load_state_dict(torch.load(args.model, weights_only=True))
    model.to(device)
    model.eval()
    print("Loaded model weights from", args.model)

calib_path = os.path.join(args.dataset_path, "calib.txt")
calib = np.genfromtxt(calib_path)
print(f"Loaded calibration from {calib_path}.")
camera_matrix, distortion_coeffs = helpers.get_camera_matrix_and_distortion_coeffs(calib)

results = {"r_error": [], "inlier_ratio": [], "gt_rot": [], "dt": []}
deg_intervals = np.array(range(0, 45, 1)) + 1
for sequence in args.sequence_names:
    # Load sequence data
    sequence_path = os.path.join(args.dataset_path, sequence)
    poses = np.genfromtxt(os.path.join(sequence_path, pose_file_name))
    print(f"Loaded poses for sequence {sequence}.")

    if args.dataset_name == "ecd":
        events = read_csv(os.path.join(sequence_path, events_file_name), header='infer', delimiter=" ", usecols=range(4)).to_numpy()
        start_time = poses[np.argmax(poses[:, 0] > events[0, 0].item() + np.max(args.model_delta_t)), 0]
        end_time = min(events[-1, 0].item(), poses[-1, 0])
    elif args.dataset_name == "eds":
        events = h5py.File(os.path.join(sequence_path, events_file_name), 'r')
        os.path.join(os.path.join(sequence_path, events_file_name))
        start_time = poses[np.argmax(poses[:, 0] > events['t'][0] * 1e-6 + np.max(args.model_delta_t)), 0]
        end_time = min(events['t'][-1] * 1e-6, poses[-1, 0])
    print(f"Loaded events for sequence {sequence}.")

    # Crop to supported shape
    max_factor_required = config["grid_size"]
    if "backbone_config" in config:
        max_factor_required = 2 ** (len(config["backbone_config"]["num_blocks"]) - 1) * \
                                config["backbone_config"]["stem"]["patch_size"] * \
                                np.max(config["backbone_config"]["stage"]["attention"]["partition_size"])
    crop = np.array(ts_shape) % max_factor_required
    crop_mask = torch.ones(ts_shape, dtype=bool)
    crop_mask[:math.ceil(crop[0] / 2)] = False
    crop_mask[:, :math.ceil(crop[1] / 2)] = False
    if crop[0] > 1:
        crop_mask[-math.floor(crop[0] / 2):] = False
    if crop[1] > 1:
        crop_mask[:, -math.floor(crop[1] / 2):] = False
    cropped_shape = [ts_shape[0] - crop[0], ts_shape[1]- crop[1]]

    if args.mcts_type == "dt":
        settings = {"shape": ts_shape, "delta_t": args.model_delta_t}
        ts_gen_type = TsGeneratorType.TimeWindow
    elif args.mcts_type == "ne":
        settings = {"shape": ts_shape, "events_per_px": args.model_events_per_px}
        ts_gen_type = TsGeneratorType.EventCount
    else:
        raise NotImplementedError(f"{args.mcts_type} is not a supported identifier for mcts_type. \
                                  Please pass the flag '--args.mcts_type' with either \
                                  'ne' (constant event count) or 'dt' (constant time window).")
    ts_gen = TsGenerator(ts_gen_type, settings=settings, device=device)

    if args.dataset_name == "ecd":
        events = torch.from_numpy(events).to(device)
    elif args.dataset_name == "eds":
        # Load timestamps into numpy array to speed-up loop
        events_t = np.array(events['t'], dtype=np.int64)

    print("Starting event loop.")
    current_event_idx = 0
    pred_list = []
    ts_vis_list = []

    with torch.inference_mode():
        for pose in tqdm(poses):
            prev_event_idx = current_event_idx

            # Feed events in batches
            if args.dataset_name == "ecd":
                current_event_idx = torch.argmax(torch.clamp(events[:, 0], max=pose[0]))
                event_batch = events[prev_event_idx:current_event_idx]
            elif args.dataset_name == "eds": 
                current_event_idx = np.argmax(events_t > pose[0] * 1e6)
                event_batch = torch.from_numpy(np.vstack([(events_t[prev_event_idx:current_event_idx] - events_t[0]) * 1e-6,
                                                        events['x'][prev_event_idx:current_event_idx],
                                                        events['y'][prev_event_idx:current_event_idx],
                                                        events['p'][prev_event_idx:current_event_idx]]).T).to(device)
            if len(event_batch) > 0:
                ts_gen.batch_update(event_batch)

            # Skip if not in experiment time range
            if pose[0] < start_time + 10.:
                continue

            # Experiment
            # ----------
            # Get time surface
            ts = ts_gen.get_ts()
            ts = ts.permute(2, 0, 1).unsqueeze(0)  # channels first
            ts = ts[..., crop_mask].reshape(list(ts.shape[:-2]) + cropped_shape)
            pred = model(ts)

            # Non-maximum-surpression
            kpts, _ = fast_nms(pred["prob"], config, top_k=500)

            # Extract descriptors
            desc_grid = pred["descriptor_grid"]
            desc = interpolate_desc_grid(desc_grid, kpts[0], cropped_shape)
            desc = desc[0].permute(1, 0).cpu().detach().numpy()
            np.savetxt(os.path.join(args.out_dir, sequence + "_descriptors.txt"), desc)

            break  # Done, one descriptor is enough
