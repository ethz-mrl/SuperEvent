import numpy as np
import cv2

def get_seq_and_idx(idx, sequence_lengths):
    seq = 0
    while idx >= sequence_lengths[seq]:
        idx -= sequence_lengths[seq]
        seq += 1
    return seq, idx

def create_kp_maps(matches, shape):
    kp_map0 = np.zeros(shape, dtype=np.float32)
    valid = matches["matches0"] > -1
    kp0 = matches["keypoints0"].astype(int)
    kp_map0[kp0[~valid][:, 1], kp0[~valid][:, 0]] = -1  # SuperGlue pseudo ground truth uses (column, row) convention
    kp_map0[kp0[valid][:, 1], kp0[valid][:, 0]] = np.arange(len(kp0[valid])) + 1

    kp_map1 = np.zeros(shape, dtype=np.float32)
    kp1 = matches["keypoints1"].astype(int)
    kp_map1[kp1[:, 1], kp1[:, 0]] = -1
    kp_map1[kp1[matches["matches0"][valid]][:, 1], kp1[matches["matches0"][valid]][:, 0]] = np.arange(len(kp0[valid])) + 1
    return kp_map0, kp_map1

def create_tencode_from_ts(ts_neg_100ms, ts_pos_100ms, delta_t):
    assert delta_t < 0.1

    # Scale to desired resolution
    scaling = 0.1 / delta_t
    e_neg = scaling * ts_neg_100ms
    e_pos = scaling * ts_pos_100ms

    # Remove events out of time delta
    e_max = np.max([e_neg, e_pos])
    e_neg = (e_neg - e_max) + 1
    e_pos = (e_pos - e_max) + 1
    e_neg = np.clip(e_neg, a_min=0., a_max=None)
    e_pos = np.clip(e_pos, a_min=0., a_max=None)

    # Invert
    e_neg[e_neg > 0] = -e_neg[e_neg > 0] + 1
    e_pos[e_pos > 0] = -e_pos[e_pos > 0] + 1

    # Remove events if another event with different polarity happened later
    e_neg[np.all([e_pos > 0, e_neg > e_pos], axis=0)] + 0.
    e_pos[np.all([e_neg > 0, e_pos > e_neg], axis=0)] + 0.

    # Create tencode
    tencode = np.zeros(list(ts_neg_100ms.shape) + [3])
    tencode[e_neg > 0] = np.vstack([np.zeros(np.sum(e_neg > 0)), 255 * e_neg[e_neg > 0], 255 * np.ones(np.sum(e_neg > 0))]).T
    tencode[e_pos > 0] = np.vstack([255 * np.ones(np.sum(e_pos > 0)), 255 * e_pos[e_pos > 0], np.zeros(np.sum(e_pos > 0))]).T

    tencode_gray = cv2.cvtColor(tencode.astype(np.uint8), cv2.COLOR_BGR2GRAY) / 255.
    return tencode_gray[None, ...].astype(np.float32)