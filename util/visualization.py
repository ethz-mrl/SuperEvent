import cv2
import math
import numpy as np

def ts2image(ts):
    #ts = np.average(ts, axis=2)  # Take average over channels
    #ts = np.uint8(255. * ts)
    #ts = cv2.cvtColor(ts, cv2.COLOR_GRAY2BGR)
    ts_out = np.ones(list(ts.shape[:2]) + [3])
    #blue_values = ts[ts[..., 2] > 0][..., 2].T
    #red_values = ts[ts[..., 7] > 0][..., 7].T
    #ts_out[ts[..., 2] > 0] = np.array([np.ones_like(blue_values), 1. - blue_values, 1. - blue_values]).T
    #ts_out[ts[..., 7] > 0] = np.array([1. - red_values, 1. - red_values, np.ones_like(red_values)]).T
    blue_values = ts[ts[..., 3] > 0][..., 3].T
    red_values = ts[ts[..., 8] > 0][..., 8].T
    ts_out[ts[..., 3] > 0] = np.array([np.ones_like(blue_values), 1. - blue_values, 1. - blue_values]).T
    ts_out[ts[..., 8] > 0] = np.array([1. - red_values, 1. - red_values, np.ones_like(red_values)]).T
    ts_out = np.rint(ts_out * 255.).astype(np.uint8)
    return ts_out

def kp_map2matches(kp_maps):
    match_list = []
    for kp_map in kp_maps:
        num_matches = round(np.max(kp_map))
        matches = [np.where(kp_map == i + 1) for i in range(num_matches)]
        match_list.append(matches)
    min_len = min([len(matches) for matches in match_list])
    match_list = [matches[:min_len] for matches in match_list]
    valid_match_mask = np.ones(min_len, dtype=bool)
    for i in range(min_len):
        for matches in match_list:
            if not len(matches[i][0]) or not len(matches[i][1]):
                valid_match_mask[i] = False
    for i in reversed(range(min_len)):
        if not valid_match_mask[i]:
            for matches in match_list:
                del matches[i]
    return match_list

def visualize_time_surface(ts):
    ts = ts2image(ts)

    cv2.imshow("Time Surface", ts)
    cv2.waitKey(0)

def visualize_matches(kp_maps, img0, img1, title="Matches", waitkey=True, return_image=False, resize_factor=1.0):
    matches = kp_map2matches(kp_maps)
    kp0 = [cv2.KeyPoint(float(resize_factor * p[1]), float(resize_factor * p[0]), 1) for p in matches[0]]
    kp1 = [cv2.KeyPoint(float(resize_factor * p[1]), float(resize_factor * p[0]), 1) for p in matches[1]]
    cv_matches = [cv2.DMatch(i, i, 0.0) for i in range(len(kp0))]

    # Add unmatched keypoints
    if np.sum(kp_maps[0] < 0) > 1:
        kp0 = kp0 + [cv2.KeyPoint(float(resize_factor * p[1]), float(resize_factor * p[0]), 1) for p in np.nonzero(kp_maps[0] < 0)]
    if np.sum(kp_maps[1] < 0) > 1:
        kp1 = kp1 + [cv2.KeyPoint(float(resize_factor * p[1]), float(resize_factor * p[0]), 1) for p in np.nonzero(kp_maps[1] < 0)]
    
    for kp in kp0:
        img0 = cv2.circle(img0, (round(kp.pt[0]), round(kp.pt[1])), radius=math.ceil(3*resize_factor), color=(17, 86, 187), thickness=math.ceil(resize_factor))
    for kp in kp1:
        img1 = cv2.circle(img1, (round(kp.pt[0]), round(kp.pt[1])), radius=math.ceil(3*resize_factor), color=(17, 86, 187), thickness=math.ceil(resize_factor))
    matched_img = cv2.drawMatches(img0, kp0, img1, kp1, cv_matches,
                                    None, matchColor=(17, 86, 187),
                                    singlePointColor=(17, 86, 187),
                                    matchesThickness=round(resize_factor),
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    if return_image:
        frames_det = np.hstack([img0, img1])
        return matched_img, frames_det
    else:
        cv2.imshow(title, matched_img)
        if waitkey:
            cv2.waitKey(0)
        return None, None

def resize_and_make_border(img, resize_factor=1.0, border_size=1):
    if border_size == 0:
        return

    # Resize
    border_size *= math.ceil(resize_factor)
    img = cv2.resize(img, (round(resize_factor * img.shape[1]), round(resize_factor * img.shape[0])),
                                interpolation=cv2.INTER_NEAREST)
    
    # Border
    img = img[border_size:-border_size, border_size:-border_size]
    img = cv2.copyMakeBorder(
        img,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )
    return img