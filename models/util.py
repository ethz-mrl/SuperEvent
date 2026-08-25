import math

import torch
from torchvision.ops import nms

def box_nms(prob, config, remove_zero=None, top_k=None, iou_threshold=0.1):
    ### Batching not supported (only implemented for batch size of 1) ###
    if len(prob.size()) == 3:
        prob = prob[0]
    # Discard detections too close to image border
    box_size_half = round(config["nms_box_size"] / 2.)
    prob[:box_size_half] = 0.
    prob[-box_size_half:] = 0.
    prob[:, :box_size_half] = 0.
    prob[:, -box_size_half:] = 0.

    # Apply NMS on detected keypoints: if remove_zero is set, this threshold will be used instead of detection threshold
    if top_k and not remove_zero:
        remove_zero = 1e-6
    if remove_zero:
        pts_candidates = (prob > remove_zero).nonzero()
        prob = prob[prob > remove_zero].flatten()
    else:
        pts_candidates = (prob > config["detection_threshold"]).nonzero()
        prob = prob[prob > config["detection_threshold"]].flatten()
    boxes_begin = pts_candidates - config["nms_box_size"] / 2.
    boxes_end = pts_candidates + config["nms_box_size"] / 2.
    boxes = torch.cat((boxes_begin, boxes_end), dim=1)
    idxs_filtered = nms(boxes, prob, iou_threshold=iou_threshold)
    if top_k:
        idxs_filtered = idxs_filtered[:top_k]
    return [pts_candidates[idxs_filtered]], [prob[idxs_filtered]]

def fast_nms(prob, config, top_k=None):
    # Find all probablilities that are the largest in their box (does not work for ties, but very unlikely to happen with floats)
    nms_mask = prob == torch.nn.functional.max_pool2d(prob, kernel_size=config["nms_box_size"], stride=1, padding=math.floor(config["nms_box_size"]/2))
    prob[~nms_mask] = 0.

    if top_k:
        det_threshold = 1e-6
    else:
        det_threshold = config["detection_threshold"]
    det_mask = prob > det_threshold
    pts_candidates = det_mask.nonzero()
    prob_candidates = prob[det_mask].flatten()

    pts_batch = [pts_candidates[pts_candidates[:, 0] == i][:, 1:] for i in range(prob.shape[0])]
    prob_batch = [prob_candidates[pts_candidates[:, 0] == i] for i in range(prob.shape[0])]

    if top_k:
        for batch in range(prob.shape[0]):
            prob_batch[batch], sorted_idxs = torch.sort(prob_batch[batch], descending=True)
            pts_batch[batch] = pts_batch[batch][sorted_idxs]
            pts_batch[batch] = pts_batch[batch][:top_k]
            prob_batch[batch] = prob_batch[batch][:top_k]

    return pts_batch, prob_batch

def pixel_to_normalized(px, shape):
    px = px.float()
    px[..., 0] = (px[..., 0] + 0.5) / shape[-2] * 2 - 1
    px[..., 1] = (px[..., 1] + 0.5) / shape[-1] * 2 - 1
    return px

def interpolate_desc_grid(desc_grid, kpts, shape):
    # Sanity checks
    while len(kpts.shape) < 4:
        kpts = kpts.unsqueeze(0)
    if len(desc_grid.shape) < 4:
        desc_grid = desc_grid.unsqueeze(0)

    # Interpolate
    kpts = pixel_to_normalized(kpts, shape)
    kpts = kpts.flip(-1)
    desc = torch.nn.functional.grid_sample(desc_grid, kpts, mode="bilinear", padding_mode="border", align_corners=False)
    desc.squeeze_(-2)
    norm = (desc * desc).sum(dim=1, keepdim=True).sqrt()
    desc = desc / (norm + 1e-6)

    return desc
