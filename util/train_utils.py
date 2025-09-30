import numpy as np
from tqdm import tqdm

import torch

from models.losses import super_event_loss

def list2device(tensor_list, device):
    if type(tensor_list).__name__ == "list":
        tensor_list = [list2device(tensor, device) for tensor in tensor_list]
    else:
        tensor_list = tensor_list.to(device)
    return tensor_list

def detach_nested_list(nested_tensor_list):
    nested_tensor_list = [[tensor.detach() for tensor in tensor_list] for tensor_list in nested_tensor_list]
    return nested_tensor_list

def val_loop(dataloader, model, device, config):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    val_loss = val_det_loss0 = val_det_loss1 = val_desc_loss = val_pos_desc_loss = val_neg_desc_loss = 0.
    num_batches = min(round(config["max_num_validation_samples"] / config["batch_size"]), int(len(dataloader)))

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for batch, data in enumerate(tqdm(dataloader, total=num_batches)):

            # Compute prediction and loss
            data = list2device(data, device)
            ts0 = data[0]
            ts1 = data[1]
            labels0 = data[2]
            labels1 = data[3]

            results_ts0 = model(ts0)
            results_ts1 = model(ts1)

            # det_loss0, det_loss1, desc_loss are only logged
            loss, det_loss0, det_loss1, desc_loss, pos_desc_loss, neg_desc_loss = super_event_loss(results_ts0["logits"], results_ts1["logits"],
                                                                                                   results_ts0["descriptors_raw"], results_ts1["descriptors_raw"],
                                                                                                   labels0, labels1, config)
            val_loss += loss
            val_det_loss0 += det_loss0
            val_det_loss1 += det_loss1
            val_desc_loss += desc_loss
            val_pos_desc_loss += pos_desc_loss
            val_neg_desc_loss += neg_desc_loss

            # Only use predefined number of validation batches
            if batch >= num_batches:
                break

    val_loss /= batch
    print(f"Val Loss: {val_loss:>8f}")
    return val_loss, \
           val_det_loss0 / batch, \
           val_det_loss1 / batch, \
           val_desc_loss / batch, \
           val_pos_desc_loss / batch, \
           val_neg_desc_loss / batch