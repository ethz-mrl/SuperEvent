import cv2
from enum import Enum
import math
import numpy as np

import torch

TsGeneratorType = Enum("TsGeneratorType", ["TimeWindow", "EventCount"])

class TsGenerator:
    def __init__(self, tsGenType, settings={}, device="cpu"):
        assert tsGenType in TsGeneratorType
        self.tsGenType = tsGenType
        ts_type_specific_key = ""
        if tsGenType == TsGeneratorType.EventCount:
            ts_type_specific_key = "events_per_px"
        elif tsGenType == TsGeneratorType.TimeWindow:
            ts_type_specific_key = "delta_t"

        # Process settings
        default_settings = {}
        default_settings["shape"] = [184, 240]
        default_settings[ts_type_specific_key] = [0.1]
        self.settings = settings
        self.device = device
        self.undistort = False

        # Sanity checks
        if "shape" not in self.settings.keys() or not len(self.settings["shape"]) == 2:
            self.settings["shape"] = default_settings["shape"]
            print("TsGenerator: Using default shape setting:", default_settings["shape"])
        else:
            self.settings["shape"] = list(self.settings["shape"])  # make sure its a list

        if ts_type_specific_key not in self.settings.keys() \
            or not np.all(np.array(self.settings[ts_type_specific_key]) > 0.):  # not all to also check empty list
            self.settings[ts_type_specific_key] = default_settings[ts_type_specific_key]
            print(f"TsGenerator: Using default {ts_type_specific_key} setting:", default_settings[ts_type_specific_key])

        if tsGenType == TsGeneratorType.EventCount:
            self.num_events_per_channel = [math.ceil(channel * self.settings["shape"][0] * self.settings["shape"][1]) for channel in self.settings["events_per_px"]]

            self.start_time = -1.0
            self.buffer_size = max(self.num_events_per_channel)

            # Initialize event state buffer
            self.buffer_events_t = -1 * torch.ones([self.buffer_size], dtype=torch.float32, device=self.device, requires_grad=False)
            self.buffer_events_x = torch.zeros([self.buffer_size], dtype=torch.int, device=self.device, requires_grad=False)
            self.buffer_events_y = torch.zeros([self.buffer_size], dtype=torch.int, device=self.device, requires_grad=False)
            self.buffer_events_p = torch.zeros([self.buffer_size], dtype=torch.int, device=self.device, requires_grad=False)

        elif tsGenType == TsGeneratorType.TimeWindow:
            self.ts_dim = torch.tensor([self.settings["delta_t"]]).reshape([-1]).to(self.device)  # ensure exactly one dim

            # Initialize time stamp tracking
            self.time_stamps = torch.zeros(self.settings["shape"] + [2, 1], dtype=torch.float32).to(self.device)

    def set_undistortion_params(self, cam_mat, dist_coeffs):
        self.undistort = True
        h_in, w_in = self.settings["shape"]
        ts_size_cv = (w_in, h_in)

        # Undistort without upscaling
        new_cam_mat, roi = cv2.getOptimalNewCameraMatrix(
            cam_mat, dist_coeffs, ts_size_cv, 1, ts_size_cv)
        
        # Remove one more pixel at each side since there can
        # still be invalid pixels from opencv rounding
        x, y, w, h = roi
        x += 1
        y += 1
        w -= 2
        h -= 2
        new_cam_mat[0, 2] -= x
        new_cam_mat[1, 2] -= y

        self.undist_maps = cv2.initUndistortRectifyMap(
            cam_mat, dist_coeffs, None, new_cam_mat, 
            (w, h), cv2.CV_16SC2)

        return new_cam_mat

    # Define Functions depending on Type
    def update(self, t, x, y, p):
        if self.tsGenType == TsGeneratorType.EventCount:
            return self.update_EventCountImpl(t, x, y, p)
        elif self.tsGenType == TsGeneratorType.TimeWindow:
            return self.update_TimeWindowImpl(t, x, y, p)
        else:
            raise NotImplementedError(f"TsGenerator with type {self.tsGenType} does not have a update()-function.")
        
    def batch_update(self, event_batch_combined_or_t, event_batch_x=None, event_batch_y=None, event_batch_p=None):
        if self.tsGenType == TsGeneratorType.EventCount:
            return self.batch_update_EventCountImpl(event_batch_combined_or_t, event_batch_x, event_batch_y, event_batch_p)
        elif self.tsGenType == TsGeneratorType.TimeWindow:
            assert event_batch_x is None and event_batch_y is None and event_batch_p is None, "Only one joint batch is supported."
            return self.batch_update_TimeWindowImpl(event_batch_combined_or_t)
        else:
            raise NotImplementedError(f"TsGenerator with type {self.tsGenType} does not have a batch_update()-function.")
        
    def is_buffer_initialized(self):
        if self.tsGenType == TsGeneratorType.EventCount:
            return self.is_buffer_initialized_EventCountImpl()
        else:
            raise NotImplementedError(f"TsGenerator with type {self.tsGenType} does not have an is_buffer_initialized()-function.")
        
    def get_ts(self):
        if self.tsGenType == TsGeneratorType.EventCount:
            ts = self.get_ts_EventCountImpl()
        elif self.tsGenType == TsGeneratorType.TimeWindow:
            ts = self.get_ts_TimeWindowImpl()
        else:
            raise NotImplementedError(f"TsGenerator with type {self.tsGenType} does not have a get_ts()-function.")
        
        if self.undistort:
            ts = cv2.remap(
                    ts.detach().cpu().numpy(), self.undist_maps[0], self.undist_maps[1], interpolation=cv2.INTER_LINEAR)
            ts = torch.from_numpy(ts).to(self.device)
        return ts


    ################################################
    ### Functions for TsGeneratorType.EventCount ###
    ################################################

    def update_EventCountImpl(self, t, x, y, p):
        event_batch_t = torch.tensor([t], device=self.device, requires_grad=False)
        event_batch_x = torch.tensor([x], device=self.device, requires_grad=False)
        event_batch_y = torch.tensor([y], device=self.device, requires_grad=False)
        event_batch_p = torch.tensor([p], device=self.device, requires_grad=False)
        self.batch_update_EventCountImpl(event_batch_t, event_batch_x, event_batch_y ,event_batch_p)

    def batch_update_EventCountImpl(self, event_batch_combined_or_t, event_batch_x=None, event_batch_y=None, event_batch_p=None):
        if event_batch_x is None or event_batch_y is None or event_batch_p is None:
            assert event_batch_x is None and event_batch_y is None and event_batch_p is None, "Only one joint batch or 4 batches of indiviual t, x, y, p are supported."

            # Convert data types
            event_batch_t = event_batch_combined_or_t[:, 0]
            event_batch_x = event_batch_combined_or_t[:, 2].to(torch.int)  # we use (row, column) instead of (x, y) coordinates
            event_batch_y = event_batch_combined_or_t[:, 1].to(torch.int)
            event_batch_p = event_batch_combined_or_t[:, 3].to(torch.int)

        else:
            event_batch_t = event_batch_combined_or_t

        num_events = len(event_batch_t)
        if num_events == 0:
            return
        if self.start_time < 0.0:
            self.start_time = event_batch_t[0]

        # Update buffer
        if num_events > self.buffer_size:
            # Batch exceeds buffer size
            self.buffer_events_t = event_batch_t[-self.buffer_size:] - self.start_time
            self.buffer_events_x = event_batch_x[-self.buffer_size:]
            self.buffer_events_y = event_batch_y[-self.buffer_size:]
            self.buffer_events_p = event_batch_p[-self.buffer_size:]
        else:
            # Normal update: First, move all old events to front...
            self.buffer_events_t[:-num_events] = self.buffer_events_t[num_events:].clone()
            self.buffer_events_x[:-num_events] = self.buffer_events_x[num_events:].clone()
            self.buffer_events_y[:-num_events] = self.buffer_events_y[num_events:].clone()
            self.buffer_events_p[:-num_events] = self.buffer_events_p[num_events:].clone()
            
            # ...then, add new events to back
            self.buffer_events_t[-num_events:] = event_batch_t - self.start_time
            self.buffer_events_x[-num_events:] = event_batch_x
            self.buffer_events_y[-num_events:] = event_batch_y
            self.buffer_events_p[-num_events:] = event_batch_p

    def is_buffer_initialized_EventCountImpl(self):
        return not torch.any(self.buffer_events_t < 0.0)

    def get_ts_EventCountImpl(self):
        valid_mask = self.buffer_events_t >= 0.0
        num_valid_events = valid_mask.sum()
        if num_valid_events < self.buffer_size: print(f"Warning (TsGenerator): Creating time surfaces with only {num_valid_events} events initialized when {self.buffer_size} events are required.")
        valid_buffer_t = self.buffer_events_t[valid_mask]
        valid_buffer_x = self.buffer_events_x[valid_mask]
        valid_buffer_y = self.buffer_events_y[valid_mask]
        valid_buffer_p = self.buffer_events_p[valid_mask]

        # Calculate start and end timestamps
        t_max = valid_buffer_t[-1]
        t_min = torch.empty([len(self.num_events_per_channel)], dtype=torch.float32, device=self.device)
        for i in range(len(self.num_events_per_channel)):
            t_min[i] = valid_buffer_t[-min(self.num_events_per_channel[i], num_valid_events)]
        t_min = t_min.repeat_interleave(2)

        # Only keep events with same x, y, p with lastest t
        sort_values = valid_buffer_x * 2 * torch.max(valid_buffer_y + 1) + valid_buffer_y * 2 + valid_buffer_p
        sort_values, sort_indeces = torch.sort(sort_values, dim=0, stable=True)  # sorted in the order of row 1, then 2, then 3, then 0

        # Every x, y, p must be different from the next one, otherwise it is repeated and has not the most recent time stamp
        # The last element alwas has the most recent time stamp for its pixel
        keep_event_mask = sort_values[:-1] != sort_values[1:]
        keep_event_mask = torch.cat([keep_event_mask, torch.tensor([True], device=self.device)])
        valid_buffer_t = valid_buffer_t[sort_indeces[keep_event_mask]]
        valid_buffer_x = valid_buffer_x[sort_indeces[keep_event_mask]]
        valid_buffer_y = valid_buffer_y[sort_indeces[keep_event_mask]]
        valid_buffer_p = valid_buffer_p[sort_indeces[keep_event_mask]]

        # Assuming event_batch contains events with [t, x, y, p] with p being an int with value 0 or 1
        ts = torch.zeros(self.settings["shape"] + [2 * len(self.num_events_per_channel)], dtype=torch.float32, device=self.device)
        for i, num_events in enumerate(self.num_events_per_channel):
            t_mask = valid_buffer_t > t_min[2*i]
            ts[valid_buffer_x[t_mask], valid_buffer_y[t_mask], 2*i+valid_buffer_p[t_mask]] = valid_buffer_t[t_mask].to(torch.float32) - t_min[2*i]
        
        # Scale between 0 and 1
        ts = ts / (t_max - t_min)

        return ts
    

    ################################################
    ### Functions for TsGeneratorType.TimeWindow ###
    ################################################
    
    def update_TimeWindowImpl(self, t, x, y, p):
        # Assuming t is float32, x and y are int, and p is an int with value 0 or 1
        self.time_stamps[x, y, p] = t

    def batch_update_TimeWindowImpl(self, event_batch):
        # Only keep events with same x, y, p with lastest t
        sort_values = event_batch[:, 1] * 2 * torch.max(event_batch[:, 2] + 1) + event_batch[:, 2] * 2 + event_batch[:, 3]
        sort_values, sort_indeces = torch.sort(sort_values, dim=0, stable=True)
        event_batch = event_batch[sort_indeces]  # The tensor is now sorted in the order of row 1, then 2, then 3, then 0

        # Every x, y, p must be different from the next one, otherwise it is repeated and has not the most recent time stamp
        # The last element alwas has the most recent time stamp for its pixel
        keep_event_mask = sort_values[:-1] != sort_values[1:]
        keep_event_mask = torch.cat([keep_event_mask, torch.tensor([True], device=self.device)])
        event_batch = event_batch[keep_event_mask]

        # Assuming event_batch contains events with [t, x, y, p] with p being an int with value 0 or 1
        t = event_batch[:, 0].float()
        x = event_batch[:, 2].int()  # we use (row, column) instead of (x, y) coordinates
        y = event_batch[:, 1].int()
        p = event_batch[:, 3].int()
        self.time_stamps[x, y, p] = t[..., None]

        # Commented out for faster runtime
        #assert not any(self.time_stamps[x, y, p] < t[..., None])

    def get_ts_TimeWindowImpl(self):
        t_max = torch.max(self.time_stamps)
        ts = self.time_stamps - t_max
        
        ts = ts + self.ts_dim
        ts = torch.clamp(ts, min=0.)
        ts = ts / self.ts_dim
        ts = torch.reshape(ts, self.settings["shape"] + [2 * len(self.ts_dim)])

        return ts
