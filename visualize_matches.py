#!/usr/bin/env python
import argparse
import cv2
from glob import glob
import math
import numpy as np
import os
import yaml

import torch
from torch.utils.data import DataLoader

from data.dataset import DataSplit, DatasetCollection
from models.super_event import SuperEvent, SuperEventFullRes
from models.util import fast_nms
from util.eval_utils import extract_keypoints_and_descriptors
from util.train_utils import list2device
from util.visualization import ts2image, visualize_matches, resize_and_make_border

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="", help="Model weights to be evaluated. If not specified, the most recent weights in saved_models/ are used.")
parser.add_argument("--config", default="config/super_event.yaml", help="Parameter configuration.")
parser.add_argument("--dataset", default="", help="Name of dataset to be used.")
parser.add_argument("--save_dir", default="", help="Save figures in this directory.")
parser.add_argument("--demo", default=False, action=argparse.BooleanOptionalAction, help="Show matches for samples with few matches in the corresponding frames.")
args = parser.parse_args()

# Flip images for some datasets
flip_img = False
if args.dataset == "ddd20" or args.dataset == "vivid":
    flip_img = True

if args.save_dir:
    os.makedirs(args.save_dir, exist_ok=True)

# Load config
with open(args.config, 'r') as f:
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
print(f"Using config:\n\n{yaml.dump(config)}")

# Set dataset if arg was set
if args.dataset:
    dataset_found = False
    for test_sequences in config["test_sequences"]:
        dataset_name = next(iter(test_sequences), "")
        if dataset_name == args.dataset:
            config["test_sequences"] = [test_sequences]
            dataset_found = True
            break
    if not dataset_found:
        raise FileNotFoundError(f"Dataset {args.dataset} does not exist in config file under {args.config}. Please specify its test sequences in the config file you provide.")

# Load model
if args.model == "":
    # Use most recent model in saved_models
    list_of_files = glob("saved_models/*.pth")
    args.model = max(list_of_files, key=os.path.getctime)

if config["pixel_wise_predictions"]:
    model = SuperEventFullRes(config)
else:
    model = SuperEvent(config)

model.load_state_dict(torch.load(args.model, weights_only=True))
model.eval()
print("Loaded model weights from", args.model)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model.to(device)

test_data = DatasetCollection(DataSplit.test, config, vis_mode=True, demo=args.demo)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

print("Showing matches on test set. Press and key for the next sample and 'q' to quit.")
with torch.inference_mode():
    for i, data in enumerate(test_dataloader):
        # Compute prediction and loss
        data_on_device = list2device(data[:4], device)
        ts0 = data_on_device[0]
        ts1 = data_on_device[1]
        labels0 = data_on_device[2]
        labels1 = data_on_device[3]

        # Convert to plottable images
        img0 = data[0][0].detach().numpy().transpose(1, 2, 0)  # channels last
        img1 = data[1][0].detach().numpy().transpose(1, 2, 0)
        img0 = ts2image(img0)
        img1 = ts2image(img1)

        if args.save_dir:
            resize_factor = 1080 / data[4].shape[1]
            frames = np.hstack([resize_and_make_border(data[4][0].detach().numpy(), resize_factor=resize_factor), resize_and_make_border(data[5][0].detach().numpy(), resize_factor=resize_factor)])
            if flip_img:
                frames = frames[::-1]
            cv2.imwrite(os.path.join(args.save_dir, data[6][0] + "_frames.jpg"), frames)
        else:
            resize_factor = 1.0
            save_frame_path = ""
        labels_img, frames_det = visualize_matches([data[2][0].detach().numpy(), data[3][0].detach().numpy()],
                           resize_and_make_border(data[4][0].detach().numpy(), resize_factor=resize_factor),
                           resize_and_make_border(data[5][0].detach().numpy(), resize_factor=resize_factor),
                           title="Ground truth matches",
                           waitkey=False, return_image=(args.save_dir != ""),
                           resize_factor=resize_factor)
        if args.save_dir:
            if flip_img:
                frames_det = frames_det[::-1]
                labels_img = labels_img[::-1]
            cv2.imwrite(os.path.join(args.save_dir, data[6][0] + "_frames_det.jpg"), frames_det)
            cv2.imwrite(os.path.join(args.save_dir, data[6][0] + "_frames_matched.jpg"), labels_img)
        print("Matches for", data[6][0])

        # Inference
        results = model(torch.cat([ts0, ts1], dim=0))

        # Non-maximum suppression
        num_kpts = int(np.max(data[2][0].detach().numpy()))
        if args.demo:
            num_kpts = results["prob"].shape[-2]
        points_nms, _ = fast_nms(results["prob"], config, top_k=num_kpts)

        desc_map0 = results["descriptors"][0].cpu().detach().numpy().transpose(1, 2, 0)  # channels last
        desc_map1 = results["descriptors"][1].cpu().detach().numpy().transpose(1, 2, 0)

        kpts0, desc0 = extract_keypoints_and_descriptors(points_nms[0].cpu().detach().numpy(), desc_map0, config["detection_threshold"], resize_factor=resize_factor)
        kpts1, desc1 = extract_keypoints_and_descriptors(points_nms[1].cpu().detach().numpy(), desc_map1, config["detection_threshold"], resize_factor=resize_factor)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        try:
            matches = bf.match(desc0, desc1)
        except cv2.error:
            matches = []

        matches = np.array(matches).tolist()
        if args.save_dir:
            img0 = resize_and_make_border(img0, resize_factor=resize_factor)
            img1 = resize_and_make_border(img1, resize_factor=resize_factor)
            ts_img = np.hstack([img0, img1])
            if flip_img:
                ts_img = ts_img[::-1]
            cv2.imwrite(os.path.join(args.save_dir, data[6][0] + "_ts.jpg"), ts_img)
        for kp in kpts0:
            img0 = cv2.circle(img0, (round(kp.pt[0]), round(kp.pt[1])), radius=math.ceil(3*resize_factor), color=(78, 83, 35), thickness=math.ceil(resize_factor))
        for kp in kpts1:
            img1 = cv2.circle(img1, (round(kp.pt[0]), round(kp.pt[1])), radius=math.ceil(3*resize_factor), color=(78, 83, 35), thickness=math.ceil(resize_factor))
        
        if args.save_dir:
            ts_det_img = np.hstack([img0, img1])
            if flip_img:
                ts_det_img = ts_det_img[::-1]
            cv2.imwrite(os.path.join(args.save_dir, data[6][0] + "_ts_det.jpg"), ts_det_img)
        matched_ts = cv2.drawMatches(img0, kpts0, img1, kpts1, matches,
                                        None, matchColor=(78, 83, 35),
                                        singlePointColor=(78, 83, 35),
                                        matchesThickness=math.ceil(resize_factor),
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        if args.save_dir:
            if flip_img:
                matched_ts = matched_ts[::-1]
            matched_frames_ts = np.concatenate((labels_img, 255 * np.ones([matched_ts.shape[0] // 25, matched_ts.shape[1], 3], dtype=np.uint8), matched_ts), axis=0)
            kp_frames_ts = np.concatenate((frames_det, 255 * np.ones([ts_det_img.shape[0] // 25, ts_det_img.shape[1], 3], dtype=np.uint8), ts_det_img), axis=0)
            raw_frames_ts = np.concatenate((frames, 255 * np.ones([ts_img.shape[0] // 25, ts_img.shape[1], 3], dtype=np.uint8), ts_img), axis=0)
            cv2.imwrite(os.path.join(args.save_dir, data[6][0] + "_ts_matched.jpg"), matched_ts)
            cv2.imwrite(os.path.join(args.save_dir, data[6][0] + "_matches.jpg"), matched_frames_ts) 
            cv2.imwrite(os.path.join(args.save_dir, data[6][0] + "_kp.jpg"), kp_frames_ts)
            cv2.imwrite(os.path.join(args.save_dir, data[6][0] + "_raw.jpg"), raw_frames_ts)
        else:
            cv2.imshow("SuperEvent matches", matched_ts)
            if (cv2.waitKey(0) & 0xFF) == ord('q'):
                cv2.destroyAllWindows()
                break

        if device != "cpu":
            torch.cuda.empty_cache()