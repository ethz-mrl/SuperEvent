# Script to generate descriptors that can be used to train DBoW2 for okvis2 (loop closure)
#
# Two sources of time surfaces are supported:
#   --source events      Replay raw events of an evaluation dataset (ecd/eds) and
#                        export the descriptors of a single time surface per sequence.
#   --source dataloader  Use the same dataloader as during training to export descriptors
#                        of many (shuffled) time surfaces of the training split.

import argparse
from glob import glob
import h5py
import hdf5plugin
import math
import numpy as np
import os
from pandas import read_csv
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import yaml

from data.dataset import DataSplit, DatasetCollection
from data.util import get_seq_and_idx
from models.super_event import SuperEvent, SuperEventFullRes
from models.util import fast_nms, interpolate_desc_grid
from ts_generation.ts_generator import TsGenerator, TsGeneratorType
from util.eval_utils import fix_seed

# Fix seed for reproducibility
fix_seed()

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--source", default="events", choices=["events", "dataloader"], help="Where the time surfaces come from. 'events' replays the raw events of an evaluation dataset, 'dataloader' uses the dataloader of the training pipeline.")
parser.add_argument("dataset_path", nargs="?", default="", help="Root directory of dataset. Only used (and required) for '--source events'.")
parser.add_argument("--dataset_name", default="", help="Dataset used for evaluation. Supported datasets are Event Camera Dataset ('ecd') and Event-aided Direct Sparse Odometry ('eds').")
parser.add_argument("--sequence_names", nargs="*", default=[], help="Names of evaluation sequences")
parser.add_argument("--config", default="config/super_event.yaml", help="Parameter configuration.")
parser.add_argument("--out_dir", default="results", help="Directory where results are saved")
parser.add_argument("--model", default="", help="Model weights to be evaluated. If not specified, the most recent weights in saved_models/ are used.")
parser.add_argument("--model_delta_t", nargs="*", default=[0.001, 0.003, 0.01, 0.03, 0.1], type=float, help="Time delta of time surfaces")
parser.add_argument("--model_events_per_px", nargs="*", default=[0.03, 0.1, 0.3, 1.0], type=float, help="Number of events per channel in time surfaces")
parser.add_argument("--mcts_type", default="ne", help="Constant number of events or constant time window duration (dt) per channel.")
parser.add_argument("--top_k", default=500, type=int, help="Number of keypoints (and therefore descriptors) extracted per time surface.")
# Arguments for '--source dataloader'
parser.add_argument("--datasets", nargs="*", default=[], help="Only for '--source dataloader'. Names of the training datasets to export descriptors for. If not specified, all datasets listed under 'dataset_names' in the config are used.")
parser.add_argument("--max_samples", default=-1, type=int, help="Only for '--source dataloader'. Maximum number of new samples exported per dataset and run. Samples already exported to 'out_dir' are skipped and do not count. -1 for all samples (default).")
args = parser.parse_args()

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
if config["pixel_wise_predictions"]:
    model = SuperEventFullRes(config)
else:
    model = SuperEvent(config)
model.load_state_dict(torch.load(args.model, weights_only=True, map_location=torch.device(device)))
model.to(device)
model.eval()
print("Loaded model weights from", args.model)

os.makedirs(args.out_dir, exist_ok=True)


def get_max_factor_required(config):
    # Shapes must be divisible by this factor to be supported by the network
    max_factor_required = config["grid_size"]
    if "backbone_config" in config:
        max_factor_required = 2 ** (len(config["backbone_config"]["num_blocks"]) - 1) * \
                                config["backbone_config"]["stem"]["patch_size"] * \
                                np.max(config["backbone_config"]["stage"]["attention"]["partition_size"])
    return max_factor_required


def extract_descriptors(ts):
    # Predict and extract the descriptors of the top_k keypoints of a single time surface
    pred = model(ts)

    # Non-maximum-surpression
    kpts, _ = fast_nms(pred["prob"], config, top_k=args.top_k)
    if len(kpts[0]) == 0:
        return np.zeros((0, config["descriptor_size"]), dtype=np.float32)

    # Extract descriptors
    desc = interpolate_desc_grid(pred["descriptor_grid"], kpts[0], ts.shape[-2:])
    return desc[0].permute(1, 0).cpu().detach().numpy()


def get_sample_identifier(data, idx):
    # Resolve the (shuffled) index of a DatasetCollection to the sequence and
    # the first time surface (image_id0) the sample originates from
    dataset_idx, sub_idx = get_seq_and_idx(data.idx_map[idx], data.dataset_lengths)
    dataset = data.dataset_list[dataset_idx]
    seq_idx, sample_idx = get_seq_and_idx(dataset.index_mapping[sub_idx], dataset.num_samples_per_seq_list)
    sequence_name = os.path.basename(os.path.normpath(dataset.sequence_paths[seq_idx]))
    image_id0 = os.path.basename(dataset.kp_seq_list[seq_idx][sample_idx]).split("_")[0]  # '<image_id0>_<image_id1>.npz'
    return sequence_name, image_id0


def export_from_dataloader():
    # Export descriptors of the time surfaces used during training
    dataset_names = args.datasets if args.datasets else config["dataset_names"]
    for dataset_name in dataset_names:
        if dataset_name not in config["dataset_names"]:
            raise ValueError(f"Dataset {dataset_name} is not listed under 'dataset_names' in {args.config}.")

    for dataset_name in dataset_names:
        # One dataset at a time so the descriptors can be saved per dataset
        dataset_config = {**config, "dataset_names": [dataset_name]}

        # Descriptors are extracted from the unwarped time surfaces
        dataset_config["homography_adaptation"] = {**config["homography_adaptation"], "enable": False}

        # Only the first time surface of a sample is used, so the matched second time surface is not loaded
        dataset_config["temporal_matching"] = {**config["temporal_matching"], "enable": False}

        # The samples of a sequence are consecutive in time and therefore have similar
        # descriptors, so shuffling is required to get a representative subset
        dataset_config["shuffle_data"] = True

        # Sequences of the same dataset may have different resolutions, so they cannot be batched
        dataset_config["batch_size"] = 1

        data = DatasetCollection(DataSplit.train, dataset_config)

        max_samples = args.max_samples if args.max_samples > 0 else len(data)

        # Resolve the output path of every sample upfront to skip the samples that were already
        # exported. Skipped samples do not count towards 'max_samples', so the requested number of
        # samples is exported as long as the dataset provides enough samples that are not exported yet.
        # Samples sharing the first time surface (temporal matching) map to the same file and are exported once.
        sample_idxs, out_paths, resolved_paths = [], [], set()
        num_skipped = 0
        for i in range(len(data)):
            if len(sample_idxs) >= max_samples:
                break

            sequence_name, image_id0 = get_sample_identifier(data, i)
            out_path = os.path.join(args.out_dir, f"{dataset_name}_{sequence_name}_{image_id0}_descriptors.txt")
            if out_path in resolved_paths or os.path.exists(out_path):
                num_skipped += 1
                continue
            resolved_paths.add(out_path)
            sample_idxs.append(i)
            out_paths.append(out_path)

        if num_skipped > 0:
            print(f"Skipping {num_skipped} samples of {dataset_name} that were already exported.")
        if args.max_samples > 0 and len(sample_idxs) < max_samples:
            print(f"Only {len(sample_idxs)} samples of {dataset_name} are left to export ({max_samples} requested).")

        dataloader = DataLoader(Subset(data, sample_idxs), batch_size=1, shuffle=False)

        num_descriptors = 0
        with torch.inference_mode():
            for sample, out_path in zip(tqdm(dataloader, desc=dataset_name), out_paths):
                # Only the first time surface of a sample is used
                ts = sample[0].to(device)

                desc = extract_descriptors(ts)
                np.savetxt(out_path, desc, fmt="%.6f")
                num_descriptors += len(desc)
        print(f"Saved {num_descriptors} descriptors of {len(sample_idxs)} samples to {args.out_dir}.")


def export_from_events():
    assert args.dataset_path, "Please provide the root directory of the dataset for '--source events'."

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
    elif args.dataset_name == "eds":
        ts_shape = [480, 640]
        pose_file_name = "stamped_groundtruth.txt"
        events_file_name = "events.h5"

    for sequence in args.sequence_names:
        # Load sequence data
        sequence_path = os.path.join(args.dataset_path, sequence)
        poses = np.genfromtxt(os.path.join(sequence_path, pose_file_name))
        print(f"Loaded poses for sequence {sequence}.")

        if args.dataset_name == "ecd":
            events = read_csv(os.path.join(sequence_path, events_file_name), header='infer', delimiter=" ", usecols=range(4)).to_numpy()
            start_time = poses[np.argmax(poses[:, 0] > events[0, 0].item() + np.max(args.model_delta_t)), 0]
        elif args.dataset_name == "eds":
            events = h5py.File(os.path.join(sequence_path, events_file_name), 'r')
            start_time = poses[np.argmax(poses[:, 0] > events['t'][0] * 1e-6 + np.max(args.model_delta_t)), 0]
        print(f"Loaded events for sequence {sequence}.")

        # Crop to supported shape
        max_factor_required = get_max_factor_required(config)
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

                desc = extract_descriptors(ts)
                out_path = os.path.join(args.out_dir, f"{args.dataset_name}_{sequence}_{pose[0]:.6f}_descriptors.txt")
                np.savetxt(out_path, desc, fmt="%.6f")
                print(f"Saved {len(desc)} descriptors to {out_path}.")

                break  # Done, one descriptor is enough


if args.source == "dataloader":
    export_from_dataloader()
else:
    export_from_events()
