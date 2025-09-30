import cv2
from enum import Enum
from glob import glob
import numpy as np
import os
import random

from torch.utils.data import Dataset

from data.homographies import sample_homography_corners, warp_points
from data.util import get_seq_and_idx, create_kp_maps, create_tencode_from_ts
from data_preparation.util.data_io import load_ts_sparse

DataSplit = Enum("DataSplit", ["train", "val", "test"])

class DatasetCollection(Dataset):
    # Remark: This Class handles shuffeling for itself to prevent dimensionality problems
    # when batching (shuffle must be set to False in DataLoader when batch_size > 1)

    def __init__(self, data_split, config, skip_to_idx=0, vis_mode=False, demo=False):
        self.skip_to_idx = skip_to_idx

        # Get dataset names
        assert data_split in DataSplit
        if data_split == DataSplit.train:
            self.dataset_names_list = config["dataset_names"]
        if data_split == DataSplit.val:
            self.dataset_names_list = [next(iter(name)) for name in config["val_sequences"]]
        if data_split == DataSplit.test:
            self.dataset_names_list = [next(iter(name)) for name in config["test_sequences"]]

        # Create sub-datasets
        self.dataset_configs_list = [{**config, "dataset_names": name} for name in self.dataset_names_list]
        self.dataset_list = [TsDataset(data_split, config, vis_mode=vis_mode, demo=demo) for config in self.dataset_configs_list]

        # Save useful data
        self.num_datasets = len(self.dataset_list)

        # Ignore elements left after batching
        self.dataset_lengths = [len(dataset) - len(dataset) % config["batch_size"] for dataset in self.dataset_list]
        self.num_elements = sum(self.dataset_lengths)
        self.num_batches_per_dataset = np.array(self.dataset_lengths) / config["batch_size"]

        # Generate random mapping of all dataset elements but with batches from same dataset
        batch_idx_arr = np.arange(sum(self.num_batches_per_dataset), dtype=int)
        
        if config["shuffle_data"]:
            np.random.shuffle(batch_idx_arr)

        # Convert random order batch indexes to sample indexes
        batch_idx_arr = batch_idx_arr * config["batch_size"]
        self.idx_map = np.concatenate([np.arange(i, i + config["batch_size"]) for i in batch_idx_arr]).astype(int)

        assert all(np.sort(self.idx_map) == np.arange(self.num_elements)), "Some elements are occurring multiple times or are missing."

        print(f"\nData statistics ({data_split.name}):")
        for i, dataset_name in enumerate(self.dataset_names_list):
            print(f"Dataset {dataset_name} contains {self.dataset_lengths[i]} samples ({100 * self.dataset_lengths[i] / self.num_elements} %).")

    def __len__(self):
        return self.num_elements
    
    def __getitem__(self, idx):
        # Prevent data I/O overhead if training is resumed
        if idx < self.skip_to_idx:
            return []
        else:
             self.skip_to_idx = 0

        idx = self.idx_map[idx]
        seq, idx = get_seq_and_idx(idx, self.dataset_lengths)
        return self.dataset_list[seq][idx]

class TsDataset(Dataset):
    def __init__(self, data_split, config, vis_mode=False, demo=False):
        self.config = config
        self.vis_mode = vis_mode

        assert data_split in DataSplit
        train_data_path = os.path.expanduser(self.config["train_data_path"])  # expand tilde
        data_base_path = os.path.join(train_data_path, data_split.name, self.config["dataset_names"])
        assert os.path.exists(data_base_path), data_base_path

        # Get paths of all sequences
        self.sequence_paths = [f.path for f in os.scandir(data_base_path) if f.is_dir()]

        # Create lists of all time surfaces
        self.ts_seq_list = [sorted(
            [p for p in glob(os.path.join(seq, "time_surfaces", "*.npz")) if "_" not in os.path.basename(p)]
            ) for seq in self.sequence_paths]
        
        # Create lists of all match and valid indeces files
        if demo:
            self.kp_seq_list = [sorted(glob(os.path.join(seq, "demo_matches", "*.npz"))) for seq in self.sequence_paths]
        else:
            self.kp_seq_list = [sorted(glob(os.path.join(seq, "sg_matches", "*.npz"))) for seq in self.sequence_paths]

        # Reduce to maximal allowed number of samples per sequence
        max_num_samples_per_sequence = -1
        if data_split == DataSplit.train and self.config["dataset_names"] in self.config["max_num_samples_per_sequence_train"].keys():
            max_num_samples_per_sequence = self.config["max_num_samples_per_sequence_train"][self.config["dataset_names"]]
        elif data_split == DataSplit.val and self.config["dataset_names"] in self.config["max_num_samples_per_sequence_val"].keys():
            max_num_samples_per_sequence = self.config["max_num_samples_per_sequence_val"][self.config["dataset_names"]]

        if max_num_samples_per_sequence > 0:
            self.kp_seq_list = [seq if len(seq) < max_num_samples_per_sequence 
                                else np.array(seq)[(np.arange(max_num_samples_per_sequence) * (len(seq) / max_num_samples_per_sequence)).astype(int).tolist()]
                                for seq in self.kp_seq_list]
        if not self.config["temporal_matching"]["enable"]:
            # Filter out duplicate time surfaces (only first ts used)
            # Filter out all occurances of the same id before the underscore
            self.kp_seq_list = [[path for i, path in enumerate(seq) 
                                 if i == 0 or os.path.basename(path).split('_')[0] != os.path.basename(seq[i - 1]).split('_')[0]]
                                for seq in self.kp_seq_list]

        # Compute sequence length
        self.num_samples_per_seq_list = [len(seq) for seq in self.kp_seq_list]
        self.num_elements = sum(self.num_samples_per_seq_list)

        # Generate random mapping of all dataset elements if desired
        self.index_mapping = np.arange(sum(self.num_samples_per_seq_list), dtype=int)
        if config["shuffle_data"]:
            np.random.shuffle(self.index_mapping)

    def __len__(self):
        return self.num_elements

    def __getitem__(self, idx):
        idx = self.index_mapping[idx]
        seq, seq_idx = get_seq_and_idx(idx, self.num_samples_per_seq_list)

        # Load matches (pseudo ground truth)
        matches = dict(np.load(self.kp_seq_list[seq][seq_idx], allow_pickle=True))

        # Time surface (input)
        ts0 = load_ts_sparse(self.ts_seq_list[seq][int(matches["image_id0"])]).astype(np.float32)
        if self.config["temporal_matching"]["enable"]:
            ts1 = load_ts_sparse(self.ts_seq_list[seq][int(matches["image_id1"])]).astype(np.float32)
        else:
            # Only use ts0 and and match with itself after homographic adaption
            ts1 = ts0.copy()
            matches["keypoints1"] = matches["keypoints0"].copy()
            matches["matches0"] = np.arange(len(matches["keypoints0"]))

        # Warp ts0 (source ts multiple times in training data -> better augumentation)
        if self.config["homography_adaptation"]["enable"]:
            ts_shape = ts0.shape[:2][::-1]
            H, _, _, _ = sample_homography_corners(ts_shape,
                                                ts_shape,
                                                difficulty=self.config["homography_adaptation"]["difficulty"],
                                                max_angle=self.config["homography_adaptation"]["max_angle"])
            ts0 = cv2.warpPerspective(ts0, H, ts_shape)
            matches["keypoints0"] = warp_points(matches["keypoints0"], H, inverse=False)
            
            # Rescale time surface if required
            if self.config["homography_adaptation"]["resize_factor"] > 0:
                ts0 = cv2.resize(ts0, (0,0), fx=self.config["homography_adaptation"]["resize_factor"],
                                fy=self.config["homography_adaptation"]["resize_factor"], interpolation=cv2.INTER_NEAREST)
                ts1 = cv2.resize(ts1, (0,0), fx=self.config["homography_adaptation"]["resize_factor"],
                                fy=self.config["homography_adaptation"]["resize_factor"], interpolation=cv2.INTER_NEAREST)
                matches["keypoints0"] = matches["keypoints0"] * self.config["homography_adaptation"]["resize_factor"]
                matches["keypoints1"] = matches["keypoints1"] * self.config["homography_adaptation"]["resize_factor"]
            matches["keypoints0"] = np.rint(matches["keypoints0"]).astype(int)
            matches["keypoints1"] = np.rint(matches["keypoints1"]).astype(int)

        if self.vis_mode:
            # Load grayscale frames
            assert not self.config["homography_adaptation"]["enable"], "Not supported."
            frame_dir = os.path.join(os.path.dirname(os.path.dirname(self.ts_seq_list[seq][int(matches["image_id0"])])), "frames")
            frame0_path = os.path.join(frame_dir, str(matches["image_id0"]) + ".png")
            frame1_path = os.path.join(frame_dir, str(matches["image_id1"]) + ".png")
            frame0 = cv2.imread(frame0_path)
            frame1 = cv2.imread(frame1_path)

            # Add string with additional information
            dataset_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(self.ts_seq_list[seq][int(matches["image_id0"])]))))
            sequence_name = os.path.basename(os.path.dirname(os.path.dirname(self.ts_seq_list[seq][int(matches["image_id0"])])))
            identifier = dataset_name + "_" + sequence_name + "_" + str(matches["image_id0"]) + "_" + str(matches["image_id1"])

        # Filter all keypoints to lie within the image bounds
        ts_shape = ts0.shape[:2][::-1]
        valid_mask_keypoints0 = np.all([matches["keypoints0"][:, 0] >= 0, matches["keypoints0"][:, 0] < ts_shape[0],
                                matches["keypoints0"][:, 1] >= 0, matches["keypoints0"][:, 1] < ts_shape[1]], axis=0)
        matches["keypoints0"] = matches["keypoints0"][valid_mask_keypoints0]
        matches["matches0"] = matches["matches0"][valid_mask_keypoints0]

        matches["keypoints1"][matches["keypoints1"][:, 0] > ts_shape[0]][:, 0] = ts_shape[0] - 1
        matches["keypoints1"][matches["keypoints1"][:, 1] > ts_shape[1]][:, 1] = ts_shape[1] - 1

        # Create keypoint maps
        kp_map0, kp_map1 = create_kp_maps(matches, ts0.shape[:2])

        # Randomly crop to be devisible grid size
        max_factor_required = self.config["grid_size"]
        if "backbone_config" in self.config:
            max_factor_required = 2 ** (len(self.config["backbone_config"]["num_blocks"]) - 1) * \
                                self.config["backbone_config"]["stem"]["patch_size"] * \
                                np.max(self.config["backbone_config"]["stage"]["attention"]["partition_size"])
        required_crop = np.array(ts0.shape)[:-1] % max_factor_required
        random_crop = np.zeros(2, dtype=int)

        random_crop[required_crop > 0] = np.random.randint(0, high=required_crop[required_crop > 0])
        if required_crop[0] > random_crop[0]:
            ts0 = ts0[random_crop[0]:random_crop[0] - required_crop[0]]
            kp_map0 = kp_map0[random_crop[0]:random_crop[0] - required_crop[0]]
        else:
            ts0 = ts0[random_crop[0]:]
            kp_map0 = kp_map0[random_crop[0]:]
        if required_crop[1] > random_crop[1]:
            ts0 = ts0[:, random_crop[1]:random_crop[1] - required_crop[1]]
            kp_map0 = kp_map0[:, random_crop[1]:random_crop[1] - required_crop[1]]
        else:
            ts0 = ts0[:, random_crop[1]:]
            kp_map0 = kp_map0[:, random_crop[1]:]

        if self.vis_mode:
            # Also crop frame0
            if required_crop[0] > random_crop[0]:
                frame0 = frame0[random_crop[0]:random_crop[0] - required_crop[0]]
            else:
                frame0 = frame0[random_crop[0]:]
            if required_crop[1] > random_crop[1]:
                frame0 = frame0[:, random_crop[1]:random_crop[1] - required_crop[1]]
            else:
                frame0 = frame0[:, random_crop[1]:]

        random_crop = np.zeros(2, dtype=int)
        random_crop[required_crop > 0] = np.random.randint(0, high=required_crop[required_crop > 0])
        if required_crop[0] > random_crop[0]:
            ts1 = ts1[random_crop[0]:random_crop[0] - required_crop[0]]
            kp_map1 = kp_map1[random_crop[0]:random_crop[0] - required_crop[0]]
        else:
            ts1 = ts1[random_crop[0]:]
            kp_map1 = kp_map1[random_crop[0]:]
        if required_crop[1] > random_crop[1]:
            ts1 = ts1[:, random_crop[1]:random_crop[1] - required_crop[1]]
            kp_map1 = kp_map1[:, random_crop[1]:random_crop[1] - required_crop[1]]
        else:
            ts1 = ts1[:, random_crop[1]:]
            kp_map1 = kp_map1[:, random_crop[1]:]

        if self.vis_mode:
            # Also crop frame1
            if required_crop[0] > random_crop[0]:
                frame1 = frame1[random_crop[0]:random_crop[0] - required_crop[0]]
            else:
                frame1 = frame1[random_crop[0]:]
            if required_crop[1] > random_crop[1]:
                frame1 = frame1[:, random_crop[1]:random_crop[1] - required_crop[1]]
            else:
                frame1 = frame1[:, random_crop[1]:]

        # Convert to tencode if required
        if self.config["input_representation"] == "tencode":
            tencode_l0 = create_tencode_from_ts(ts0[..., 4], ts0[..., 9], 0.01)  # value only for ablation, 0.02 in paper
            tencode_m0 = create_tencode_from_ts(ts0[..., 4], ts0[..., 9], random.uniform(0.02, 0.035))
            tencode_h0 = create_tencode_from_ts(ts0[..., 4], ts0[..., 9], random.uniform(0.035, 0.05))
            tencode_l1 = create_tencode_from_ts(ts1[..., 4], ts1[..., 9], 0.01)  # value only for ablation, 0.02 in paper

            # Weird order is for capability with returned time surfaces when used for inference 
            return tencode_l0, tencode_l1, kp_map0, kp_map1, tencode_m0, tencode_h0

        # Change from channels last to channels first
        ts0 = ts0.transpose(2, 0, 1)
        ts1 = ts1.transpose(2, 0, 1)

        if self.config["input_representation"] == "ts":
            ts0 = np.max([ts0[2], ts0[7]], axis=0)[None]
            ts1 = np.max([ts1[2], ts1[7]], axis=0)[None]
        elif self.config["input_representation"] == "mcts_1":
            ts0 = ts0[[2, 7]]
            ts1 = ts1[[2, 7]]

        if self.vis_mode:
            return ts0, ts1, kp_map0, kp_map1, frame0, frame1, identifier

        return ts0, ts1, kp_map0, kp_map1