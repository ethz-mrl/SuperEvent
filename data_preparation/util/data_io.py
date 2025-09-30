import cv2
from enum import Enum
import glob
import h5py
import numpy as np
import os
from pandas import read_csv
from scipy.sparse import csr_matrix

RequiredData = Enum("DataSplit", ["events", "event_time_range", "images", "image_stamps", "calib"])

# ----- High level dataset loading -----
def load_dataset(dataset, path, required_data):
    assert required_data

    # Event time range will be only computed if not all events are required
    if RequiredData.events in required_data:
        only_event_time_range = False
        if RequiredData.event_time_range in required_data:
            required_data.remove(RequiredData.event_time_range)
    if RequiredData.event_time_range in required_data:
        only_event_time_range = True
        required_data.append(RequiredData.events)

    supported_datasets = ["ecd", "fpv", "mvsec", "vivid", "ddd20", "griffin", "hdr"]
    events = images = image_stamps = calib = 0
    if dataset == supported_datasets[0] or \
       dataset == supported_datasets[3] or \
       dataset == supported_datasets[5]:  # ecd, converted rosbags (vivid, griffin)
        if RequiredData.events in required_data:
            events = load_events_txt(os.path.join(path, "events.txt"), only_event_time_range)
        if RequiredData.images in required_data: 
            images = load_images_ecd(os.path.join(path, "images"))
        if RequiredData.image_stamps in required_data:
            image_stamps = load_image_timestamps_ecd(os.path.join(path, "images.txt"))
        if RequiredData.calib in required_data:
            calib = load_calib_txt(os.path.join(path, "calib.txt"))
    elif dataset == supported_datasets[1]:  # fpv
        if RequiredData.events in required_data:
            events = load_events_txt(os.path.join(path, "events.txt"), only_event_time_range)
        if RequiredData.images in required_data or RequiredData.image_stamps in required_data:
            images, image_stamps = load_images_and_stamps_fpv(os.path.join(path, "images.txt"))
        if RequiredData.calib in required_data:
            calib = load_calib_txt(os.path.join(path, "calib.txt"))
    elif dataset == supported_datasets[2]:  # mvsec
        if RequiredData.events in required_data or RequiredData.images in required_data or RequiredData.image_stamps in required_data:
            hdf5_file_list = glob.glob(os.path.join(path, "*.hdf5"))
            assert len(hdf5_file_list) == 1, "There should only be one .hdf5 file in folder " + path
            events, images, image_stamps = load_data_mvsec(hdf5_file_list[0], only_event_time_range)
        if RequiredData.calib in required_data:
            calib = load_calib_txt(os.path.join(path, "calib.txt"))
    elif dataset == supported_datasets[4]:  # ddd20
        if RequiredData.events in required_data or RequiredData.images in required_data or RequiredData.image_stamps in required_data:
            hdf5_file_list = glob.glob(os.path.join(path, "*.hdf5.exported.hdf5"))  # data must be exported first
            assert len(hdf5_file_list) == 1, "There should only be one .exported.hdf5 file in folder " + path
            events, images, image_stamps = load_data_ddd20(hdf5_file_list[0], only_event_time_range)
        if RequiredData.calib in required_data:
            calib = load_calib_txt(os.path.join(path, "calib.txt"))
    elif dataset == supported_datasets[6]:  # hdr
        mat_file_list = glob.glob(os.path.join(path, "*.mat"))
        assert len(mat_file_list) == 1, "There should only be one .mat file in folder " + path
        events, images, image_stamps, calib = load_data_hdr(mat_file_list[0], only_event_time_range)
    else:
        raise NotImplementedError("Dataset", dataset, " is not supported. Supported datasets:", supported_datasets)
    
    # Shift time to start with first event
    if RequiredData.events in required_data and RequiredData.event_time_range not in required_data and RequiredData.image_stamps in required_data:
        start_time = events[0, 0]
        events[:, 0] = events[:, 0] - start_time
        image_stamps = image_stamps - start_time
    
    # Sanity checks
    if RequiredData.event_time_range in required_data:
        assert events.shape == (2,)
        assert events[0] < events[1], str(events)
    elif RequiredData.events in required_data:  # this is the setting to load all events (without RequiredData.event_time_range)
        assert events.shape[1] == 4
    if RequiredData.images in required_data:
        assert len(images) > 0
    if RequiredData.image_stamps in required_data:
        assert len(image_stamps.shape) == 1
    if RequiredData.calib in required_data:
        assert len(calib) == 9 or len(calib) == 8

    return events, images, image_stamps, calib

def load_dataset_images_calib(dataset, path):
    supported_datasets = ["ecd", "fpv"]
    if dataset == supported_datasets[0]:  # ecd
        images = load_images_ecd(os.path.join(path, "images"))
        calib = load_calib_txt(os.path.join(path, "calib.txt"))
    elif dataset == supported_datasets[1]:  # fpv
        images, _ = load_images_and_stamps_fpv(os.path.join(path, "images.txt"))
        calib = load_calib_txt(os.path.join(path, "calib.txt"))
    else:
        raise NotImplementedError("Dataset", dataset, " is not supported. Supported datasets:", supported_datasets)
    
    # Sanity checks
    assert len(calib) == 9 or len(calib) == 8

    return images, calib


# ----- General .txt files -----
def load_calib_txt(path):
    calib_data = np.genfromtxt(path)
    print("Calibration loaded")
    return calib_data

def load_events_txt(path, only_event_time_range=False):
    print("Loading events from", path)

    if only_event_time_range:
        num_lines = sum(1 for _ in open(path))
        events = read_csv(path, header='infer', delimiter=" ", usecols=[0], skiprows=range(2, num_lines - 1)).to_numpy()
        events = events.flatten()
        print("Events in time range", events)
    else:  # load all events
        events = read_csv(path, header='infer', delimiter=" ", usecols=range(4)).to_numpy()
    
    print("Raw events loaded")
    return events


# ----- Event camera dataset ------
def load_images_ecd(path):
    print("Loading images from" , path, "...")
    images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in sorted(glob.glob(os.path.join(path, '*.png')))]
    print("Loaded", len(images), "images.")
    return images

def load_image_timestamps_ecd(path):
    timestamps = np.genfromtxt(path, usecols=[0])
    print("Loaded image timestamps")
    return timestamps


# ------ RPG FPV drone racing dataset -----
def load_images_and_stamps_fpv(path):
    # Load data from images.txt
    print("Loading data from", path)
    data = read_csv(path, header='infer', delimiter=" ")
    data_keys = data.keys()
    image_timestamps = data[data_keys[1]].to_numpy()
    image_path_list = data[data_keys[2]]
    
    # Load images
    images = [cv2.imread(os.path.join(os.path.dirname(path), img_path), cv2.IMREAD_GRAYSCALE) for img_path in image_path_list]
    print("Loaded", len(images), "images.")
    return images, image_timestamps

# ------ MVSEC h5-file -----
def load_data_mvsec(path, only_event_time_range=False):
    assert os.path.isfile(path)

    data = h5py.File(path, "r")["davis"]["left"]

    if only_event_time_range:
        events = np.array([data["events"][0][2], data["events"][-1][2]])
    else:  # load all events
        events = np.array(data["events"])
        events = events[:, [2, 0, 1, 3]]  # fix order: t, x, y, p
        events[:, -1] = np.clip(events[:, -1], 0, 1)

    images = [np.array(img) for img in data["image_raw"]]
    image_timestamps = np.array(data["image_raw_ts"])

    return events, images, image_timestamps

# ------ DDD20 exported h5-file -----
def load_data_ddd20(path, only_event_time_range=False):
    assert os.path.isfile(path)

    data = h5py.File(path, "r")

    # Detect if data is partially corrupted
    idx_max_event_timestamp = np.argmax(data["event"][:, 0])
    if idx_max_event_timestamp + 1 < len(data["event"]):
        print("Warning: Sequence", path, "seems to have partially corrupted timestamps. Only the first", (idx_max_event_timestamp + 1) / len(data["event"]) * 100, "% of the data can be used.")
        if only_event_time_range:
            events = np.array([data["event"][0][0], data["event"][idx_max_event_timestamp][0]]) * 1e-6
        else:
            events = np.array(data["event"][:idx_max_event_timestamp + 1], dtype=np.float32)
            events[:, 0] = events[:, 0] * 1e-6  # convert event timestamp to s

        idx_max_frame_timestamp = np.argmax(data["frame_ts"])
        images = [np.array(img) for img in data["frame"][:idx_max_frame_timestamp + 1]]
        image_timestamps = np.array(data["frame_ts"][:idx_max_frame_timestamp + 1]).reshape([-1])

    else:
        # Timestamps not corrupted
        if only_event_time_range:
            events = np.array([data["event"][0][0], data["event"][-1][0]]) * 1e-6 
        else:  # load all events
            events = np.array(data["event"], dtype=np.float32)
            events[:, 0] = events[:, 0] * 1e-6  # convert event timestamp to s

        images = [np.array(img) for img in data["frame"]]
        image_timestamps = np.array(data["frame_ts"]).reshape([-1])

    return events, images, image_timestamps

# ------ HDR exported h5-file -----
def load_data_hdr(path, only_event_time_range=False):
    assert os.path.isfile(path)

    data = h5py.File(path, "r")

    if only_event_time_range:
        events = np.array([data["events"][0][0], data["event"][0][-1]], dtype=np.float32) * 1e-6 
    else:  # load all events
        events = np.array(data["events"], dtype=np.float32).transpose()
        events[:, 0] = events[:, 0] * 1e-6  # convert event timestamp to s
        # Fix matlab indexing
        events[:, 1] = events[:, 1] - 1
        events[:, 2] = events[:, 2] - 1

    images = [np.array(img).transpose() for img in data["image"]]
    image_timestamps = np.array(data["time_image"]).reshape([-1]) * 1e-6

    calib = np.array([1., 1., 0., 0., 0., 0., 0., 0., 0.])  # Skip undistortion

    return events, images, image_timestamps, calib

# ----- Save time surfaces sparse and compressed ----- #
def save_ts_sparse(path, np_arr):
    # Convert to sparse matrix
    sparse_arr_list = [csr_matrix(np_arr[:, :, i]) for i in range(np_arr.shape[2])]

    # Save properties in lists
    data = np.array([csr.data for csr in sparse_arr_list], dtype=object)
    indices = np.array([csr.indices for csr in sparse_arr_list], dtype=object)
    indptr = np.array([csr.indptr for csr in sparse_arr_list], dtype=object)
    shapes = np.array([csr.shape for csr in sparse_arr_list], dtype=object)

    # Save
    np.savez_compressed(path, data=data, indices=indices, indptr=indptr, shapes=shapes)

# ----- Load time surfaces sparse and compressed ----- #
def load_ts_sparse(path):
    # Load
    loaded = np.load(path, allow_pickle=True)

    # Reconstruct CSR
    loaded_sparse_arr = [csr_matrix((loaded['data'][i], loaded['indices'][i], loaded['indptr'][i]), shape=loaded['shapes'][i]) for i in range(len(loaded['data']))]


    # Convert to np.ndarray
    loaded_sparse_arr = [arr.astype(float).toarray() for arr in loaded_sparse_arr]
    loaded_arr = np.stack(loaded_sparse_arr, axis=2)

    return loaded_arr
