#!/usr/bin/env python
import argparse
import numpy as np
import os
import sys
import time
import yaml

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.dataset import DataSplit, DatasetCollection
from models.losses import super_event_loss
from models.super_event import SuperEvent, SuperEventFullRes
from util.train_utils import list2device, val_loop
from util.eval_utils import fix_seed
torch.autograd.set_detect_anomaly(True)

# Fix slurm output
sys.stdout.reconfigure(line_buffering=True, write_through=True)

# Fix seed for reproducibility
fix_seed()

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config/super_event.yaml", help="Parameter configuration.")
parser.add_argument("--model", default="", help="Pretrained model weights")
parser.add_argument("--slurm_id", default="")
args = parser.parse_args()

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

# Create experiment name: config_name + timestamp
config_name = os.path.splitext(os.path.basename(args.config))[0]
exp_name = config_name + "_" + time.strftime("%Y%m%d-%H%M%S")

# If slurm_id is specified, check if training with same ID exists
resume_training = False
if args.slurm_id:
    slurm_path = os.path.join("slurm", args.slurm_id)
    slurm_config_backup_path = os.path.join(slurm_path, "config.yaml")
    if os.path.isdir(slurm_path):
        # Load config and model
        with open(slurm_config_backup_path, "r") as f:
            config = yaml.safe_load(f)
        exp_name = config["exp_name"]
        resume_training = True

    else:
        os.makedirs(slurm_path)

        # Save config with model path
        config["exp_name"] = exp_name
        with open(slurm_config_backup_path, 'w') as f:
            yaml.dump(config, f)

print(f"Using config:\n\n{yaml.dump(config)}")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

if config["pixel_wise_predictions"]:
    model = SuperEventFullRes(config).to(device)
else:
    model = SuperEvent(config).to(device)

# Initialize logger and model_path
print("Experiment name:", exp_name)
model_weights_path = os.path.join("saved_models", exp_name + ".pth")
writer = SummaryWriter(log_dir=os.path.join("runs", exp_name))
os.makedirs(os.path.dirname(model_weights_path), exist_ok=True)

# Load pretrained weights if provided
if not args.model == "":
    model.load_state_dict(torch.load(args.model, weights_only=True))

# Init optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

# Some variables for training
best_val_loss = 1e6
best_epoch = 0
skip_to_idx = 0

if resume_training:
    train_checkpoint = torch.load(os.path.join(slurm_path, "train_checkpoint.pt"), weights_only=True)
    model.load_state_dict(train_checkpoint["model_state_dict"])
    optimizer.load_state_dict(train_checkpoint["optimizer_state_dict"])
    best_val_loss = train_checkpoint["best_val_loss"]
    best_epoch = train_checkpoint["best_epoch"]
    skip_to_idx = train_checkpoint["batch"] * config["batch_size"]

# Init data loaders
train_data = DatasetCollection(DataSplit.train, config, skip_to_idx=skip_to_idx)
val_data = DatasetCollection(DataSplit.val, config)
train_dataloader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=False)
val_dataloader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=False)

# Constants for training
epochs = config["epochs"]
num_samples = len(train_dataloader.dataset)
num_train_batches_before_validation = round(config["num_train_samples_before_validation"] / config["batch_size"])

for t in range(epochs):

    # Skip to current epoch when training is resumed
    if resume_training and train_checkpoint["epoch"] > t:
        continue

    print(f"\nEpoch {t}\n-------------------------------")
    train_total_loss = train_det_loss0 = train_det_loss1 = train_desc_loss = train_pos_desc_loss = train_neg_desc_loss = 0

    for batch, data in enumerate(train_dataloader):
        # Skip to current iteration when training is resumed
        if resume_training and train_checkpoint["batch"] > batch:
            continue
        else:
            resume_training = False

        # Set training mode
        model.train()

        # Channel dropout
        if config["ts_dropout"]:
            dropout_mask = [False]
            while not any(dropout_mask):  # prevent dropout on all channels
                dropout_mask = np.random.random_sample(size=5) < config["ts_dropout"]
            dropout_mask = np.hstack([dropout_mask, dropout_mask])
            data[0][:, dropout_mask] = 0.
            data[1][:, dropout_mask] = 0.

        # Compute prediction and loss
        data = list2device(data, device)
        ts0 = data[0]
        ts1 = data[1]
        labels0 = data[2]
        labels1 = data[3]

        # Forward pass
        results_ts0 = model(ts0)
        results_ts1 = model(ts1)

        # det_loss0, det_loss1, desc_loss are only logged
        loss, det_loss0, det_loss1, desc_loss, pos_desc_loss, neg_desc_loss = super_event_loss(results_ts0["logits"], results_ts1["logits"],
                                                                                               results_ts0["descriptors_raw"], results_ts1["descriptors_raw"],
                                                                                               labels0, labels1, config)
        train_total_loss += loss
        train_det_loss0 += det_loss0
        train_det_loss1 += det_loss1
        train_desc_loss += desc_loss
        train_pos_desc_loss += pos_desc_loss
        train_neg_desc_loss += neg_desc_loss

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Log train loss
        if batch % config["num_batches_train_logging"] == 0 and batch > 0:
            print(f"loss: {train_total_loss / config["num_batches_train_logging"]:>7f}  [{batch * config["batch_size"]:>5d}/{num_samples:>5d}]")

            # Tensorboard logging (start at 0)
            it = t * num_samples + config["batch_size"] * (batch - config["num_batches_train_logging"])
            writer.add_scalar("train/loss", train_total_loss / config["num_batches_train_logging"], it)
            writer.add_scalar("train/loss_det0", train_det_loss0 / config["num_batches_train_logging"], it)
            writer.add_scalar("train/loss_det1", train_det_loss1 / config["num_batches_train_logging"], it)
            writer.add_scalar("train/loss_desc", train_desc_loss / config["num_batches_train_logging"], it)
            writer.add_scalar("train/pos_desc_loss", train_pos_desc_loss / config["num_batches_train_logging"], it)
            writer.add_scalar("train/neg_desc_loss", train_neg_desc_loss / config["num_batches_train_logging"], it)

            # Reset train losses
            train_total_loss = train_det_loss0 = train_det_loss1 = train_desc_loss = train_pos_desc_loss = train_neg_desc_loss = 0.

            # Save checkpoint to resume training if SLURM terminates job
            if args.slurm_id:
                batch_idx = batch + 1
                epoch_idx = t
                if batch_idx == len(train_dataloader):  # last batch in epoch
                    batch_idx = 0
                    epoch_idx = t + 1
                torch.save({
                    "epoch": epoch_idx,
                    "batch": batch_idx,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch
                    }, os.path.join(slurm_path, "train_checkpoint.pt"))

        # Validate after specified number of iterations
        if batch % num_train_batches_before_validation == 0 and t + batch > 0:
            val_loss, val_det_loss0, val_det_loss1, val_desc_loss, val_pos_desc_loss, val_neg_desc_loss = val_loop(val_dataloader, model, device, config)

            # Tensorboard logging (start at 0)
            it = t * num_samples + config["batch_size"] * (batch - num_train_batches_before_validation)
            writer.add_scalar("val/loss", val_loss, it)
            writer.add_scalar("val/loss_det0", val_det_loss0, it)
            writer.add_scalar("val/loss_det1", val_det_loss1, it)
            writer.add_scalar("val/loss_desc", val_desc_loss, it)
            writer.add_scalar("val/pos_loss_desc", val_pos_desc_loss, it)
            writer.add_scalar("val/neg_loss_desc", val_neg_desc_loss, it)

            # Check if this was the best validation loss so far
            if val_loss < best_val_loss:
                best_epoch = t + batch * config["batch_size"] / num_samples
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_weights_path)
                print(f"Model weights from epoch {best_epoch} saved in", model_weights_path)

print("Training done.")
