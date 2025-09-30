import random

import torch
from torchvision.ops import sigmoid_focal_loss

def double_softmax_distance(desc_0, desc_1, temperature=1.0):
    # Tahen from https://github.com/facebookresearch/silk/blob/main/lib/matching/mnn.py#L36
    similarity = torch.matmul(desc_0, desc_1.T) / temperature
    matching_probability = torch.softmax(similarity, dim=0) * torch.softmax(
        similarity, dim=1
    )
    return 1.0 - matching_probability

def detector_loss(logits, ground_truth_keypoint_map, config):
    # Make keypoints with label -1 positive so they can be used for training
    ground_truth_keypoint_map = torch.abs(ground_truth_keypoint_map)

    if config["pixel_wise_predictions"]:
        labels = ground_truth_keypoint_map.clamp(max=1.)  # Make binary
        loss = sigmoid_focal_loss(logits, labels, alpha=config["detector_loss_alpha"],
                                  gamma=config["detector_loss_gamma"], reduction="mean")
        return loss

    else:  # original SuperPoint loss
        # Convert the boolean keypoint map to indices + "no interest point" dustbin
        labels = ground_truth_keypoint_map[:, None, :, :]
        labels = torch.nn.functional.pixel_unshuffle(labels, config["grid_size"])
        shape = list(labels.shape)
        shape[1] = 1  # Set channel size to 1
        labels = torch.cat((2*labels, torch.ones(shape, device=labels.device)), dim=1)  # add "no interest point" dustbin
        # Add a small random matrix to break ties in argmax
        labels = torch.argmax(labels + 0.1 * torch.rand(labels.shape, device=labels.device), dim=1)

        loss = torch.nn.CrossEntropyLoss()
        return loss(logits, labels)

def descriptor_loss(descriptors0, descriptors1, keypoint_map0, keypoint_map1, config):
    # Corresponding keypoints should have an descriptor distance of 0 while all other combinations
    # should have a descriptor distance of 1
    # Grid cells that do not contain a desciptor should be not used for loss calculation
    
    # Reduce keypoint map resolution to descriptor map resolution
    batch_size, _, Hc, Wc, = descriptors0.shape
    if config["pixel_wise_predictions"]:
        Hc = Hc // config["grid_size"]
        Wc = Wc // config["grid_size"]

    # Filter out unmatched keypoints (label is -1) and  add channel: shape is [B, C, H, W]
    keypoint_map0 = keypoint_map0.clamp(min=0.)[:, None, :, :]
    keypoint_map1 = keypoint_map1.clamp(min=0.)[:, None, :, :]

    # Get feature with highest ID if there are multiple in the same cell: shape is [B, C, Hc, Wc]
    keypoint_map_grid0, kp_indeces0 = torch.nn.functional.max_pool2d(keypoint_map0, config["grid_size"], stride=config["grid_size"], return_indices=True)
    keypoint_map_grid1, kp_indeces1 = torch.nn.functional.max_pool2d(keypoint_map1, config["grid_size"], stride=config["grid_size"], return_indices=True)

    # Select the descriptors at the keypoints
    if config["pixel_wise_predictions"]:
        descriptors0 = descriptors0.flatten(start_dim=2)[:, :, kp_indeces0.flatten()].reshape((batch_size, -1, Hc, Wc))
        descriptors1 = descriptors1.flatten(start_dim=2)[:, :, kp_indeces1.flatten()].reshape((batch_size, -1, Hc, Wc))

    # Get difference between two grids. 0 difference means same index (no features will be filtered out
    # by vaid mask)
    keypoint_map_grid0 = torch.reshape(keypoint_map_grid0, (batch_size, Hc, Wc, 1, 1))
    keypoint_map_grid1 = torch.reshape(keypoint_map_grid1, (batch_size, 1, 1, Hc, Wc))
    kp_correspondence = keypoint_map_grid0 - keypoint_map_grid1
    s = torch.isclose(kp_correspondence, torch.tensor([0.], device=kp_correspondence.device)).float()
    # s[id_batch, h, w, h', w'] == 1 if the point of coordinates (h, w) corresponds to
    # point (h', w') and 0 otherwise

    # Create keypoint map to filter out invalid descriptors
    keypoint_grid_mask0 = keypoint_map_grid0.clamp(max=1.)
    keypoint_grid_mask1 = keypoint_map_grid1.clamp(max=1.)
    merged_keypoint_grid_mask = keypoint_grid_mask0 * keypoint_grid_mask1

    # Normalize the descriptors and compute their pairwise dot product
    descriptors0 = descriptors0[:, :, :, :, None, None]
    descriptors1 = descriptors1[:, :, None, None, :, :]
    descriptors0 = torch.nn.functional.normalize(descriptors0, dim=1)
    descriptors1 = torch.nn.functional.normalize(descriptors1, dim=1)
    dot_product_desc = torch.sum(descriptors0 * descriptors1, dim=1)
    dot_product_desc = torch.nn.functional.relu(dot_product_desc)
    # Normalize over image grids
    dot_product_desc = torch.reshape(
        torch.nn.functional.normalize(
            torch.reshape(dot_product_desc, (batch_size, Hc, Wc, Hc*Wc))
            , dim=3)
        , (batch_size, Hc, Wc, Hc, Wc))
    dot_product_desc = torch.reshape(
        torch.nn.functional.normalize(
            torch.reshape(dot_product_desc, (batch_size, Hc*Wc, Hc, Wc))
            , dim=1)
        , (batch_size, Hc, Wc, Hc, Wc))
    # dot_product_desc[id_batch, h, w, h', w'] is the dot product between the
    # descriptor at position (h, w) in the original descriptors map and the
    # descriptor at position (h', w') in the second image

    # Compute the loss (filter out descriptors without keypoints in pseudo gt)
    positive_dist = (config["positive_margin"] - dot_product_desc).clamp(min=0.) * merged_keypoint_grid_mask
    negative_dist = (dot_product_desc - config["negative_margin"]).clamp(min=0.) * merged_keypoint_grid_mask
    loss = config["lambda_d"] * s * positive_dist + (1 - s) * negative_dist
    normalization = (torch.sum(keypoint_grid_mask0)).clamp(min=1.)
    loss = torch.sum(loss / normalization)

    # Only for tuning and debugging
    with torch.no_grad():
        pos_loss = torch.sum(config["lambda_d"] * s * positive_dist ) / normalization
        neg_loss = torch.sum((1 - s) * negative_dist ) / normalization
    
    return loss, pos_loss, neg_loss

def double_softmax_loss(descriptors0, descriptors1, keypoint_map0, keypoint_map1, config):
    max_num_keypoints = 500  # probably never reached, just to prevent OOM
    with torch.no_grad():
        batch_size = descriptors0.shape[0]

        # Count matches for each sample
        num_matches_per_sample0 = torch.abs(keypoint_map0.clamp(min=-1, max=1)).sum(dim=(1,2), dtype=int).clamp(max=max_num_keypoints)
        num_matches_per_sample1 = torch.abs(keypoint_map1.clamp(min=-1, max=1)).sum(dim=(1,2), dtype=int).clamp(max=max_num_keypoints)
        cum_matches_until_sample0 = torch.cumsum(torch.cat((torch.tensor([0], device=num_matches_per_sample0.device), num_matches_per_sample0)), dim=0)
        cum_matches_until_sample1 = torch.cumsum(torch.cat((torch.tensor([0], device=num_matches_per_sample1.device), num_matches_per_sample1)), dim=0)

        # Sanity check
        if torch.max(num_matches_per_sample0) > max_num_keypoints or torch.max(num_matches_per_sample1) > max_num_keypoints:
            print(f"Not all keypoints will be used for evaluation: {torch.max(num_matches_per_sample0)} or {torch.max(num_matches_per_sample1)} exceeds threshold of {max_num_keypoints} keypoints.")
                                             
        # Get indexes of keypoint labels
        kp_idxs0 = torch.nonzero(keypoint_map0, as_tuple=True)
        kp_idxs1 = torch.nonzero(keypoint_map1, as_tuple=True)

        # Only select pixels with keypoints in pseudo gt
        match_indeces0 = keypoint_map0[kp_idxs0]
        match_indeces1 = keypoint_map1[kp_idxs1]

        # Set negative match indeces (= no match in pseudo gt) to nan so they are not matched
        match_indeces0[match_indeces0<0] = torch.nan
        match_indeces1[match_indeces1<1] = torch.nan

        # Create tensor with constant shape
        batched_match_idxs0 = torch.nan * torch.ones(batch_size, max_num_keypoints, device=match_indeces0.device)
        batched_match_idxs1 = torch.nan * torch.ones(batch_size, max_num_keypoints, device=match_indeces1.device)
        for batch in range(batch_size):
            batched_match_idxs0[batch, :num_matches_per_sample0[batch]] = match_indeces0[cum_matches_until_sample0[batch]:cum_matches_until_sample0[batch+1]]
            batched_match_idxs1[batch, :num_matches_per_sample1[batch]] = match_indeces1[cum_matches_until_sample1[batch]:cum_matches_until_sample1[batch+1]]

        # Get difference between two grids. 0 difference means same index (no features will be filtered out
        # by vaid mask)
        batched_match_idxs0 = torch.reshape(batched_match_idxs0, (batch_size, max_num_keypoints, 1))
        batched_match_idxs1 = torch.reshape(batched_match_idxs1, (batch_size, 1, max_num_keypoints))
        match_correspondence = batched_match_idxs0 - batched_match_idxs1
        s_idxs = torch.nonzero(match_correspondence == 0, as_tuple=True)
        # s[id_batch, h, w, h', w'] == 1 if the point of coordinates (h, w) corresponds to
        # point (h', w') and 0 otherwise

    # Normalize and filter descriptors to only use the ones at keypoints
    descriptors0 = torch.nn.functional.normalize(descriptors0, dim=1)
    descriptors1 = torch.nn.functional.normalize(descriptors1, dim=1)
    eval_descriptors0 = descriptors0.permute(0, 2, 3, 1)[kp_idxs0]
    eval_descriptors1 = descriptors1.permute(0, 2, 3, 1)[kp_idxs1]

    # Calculate loss
    loss_batch = [double_softmax_distance(eval_descriptors0[cum_matches_until_sample0[batch]:cum_matches_until_sample0[batch+1]],
                                      eval_descriptors1[cum_matches_until_sample1[batch]:cum_matches_until_sample1[batch+1]],
                                      config["double_softmax_loss_temperature"])
                                      [s_idxs[1][s_idxs[0]==batch], s_idxs[2][s_idxs[0]==batch]]  # select roundtrips in the tensor of distances
                    for batch in range(batch_size)]
    normalization = max(len(s_idxs[0]), 1)
    return torch.sum(torch.cat(loss_batch)) / normalization

def super_event_loss(logits0, logits1, descriptors0, descriptors1, ground_truth_keypoint_map0, ground_truth_keypoint_map1, config):
    det_loss0 = detector_loss(logits0, ground_truth_keypoint_map0, config)
    det_loss1 = detector_loss(logits1, ground_truth_keypoint_map1, config)
    if config["pixel_wise_predictions"]:
        desc_loss = double_softmax_loss(descriptors0, descriptors1, ground_truth_keypoint_map0, ground_truth_keypoint_map1, config)
        pos_loss = neg_loss = 0
    else:
        desc_loss, pos_loss, neg_loss = descriptor_loss(descriptors0, descriptors1, ground_truth_keypoint_map0, ground_truth_keypoint_map1, config)

    loss = (det_loss0 + det_loss1 + config["lambda_loss"] * desc_loss)
    return loss, det_loss0, det_loss1, config["lambda_loss"] * desc_loss, config["lambda_loss"] * pos_loss, config["lambda_loss"] * neg_loss