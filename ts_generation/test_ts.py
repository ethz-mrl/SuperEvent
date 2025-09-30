from time import time
import torch

from util.eval_utils import fix_seed
from generate_ts import TsGenerator

def test_ts():
    # Test incorrect input
    ts_gen = TsGenerator(settings={"shape": [184], "delta_t": -0.01})
    assert len(ts_gen.settings["shape"]) == 2
    assert torch.all(ts_gen.ts_dim > 0)

    # Test fisheye lens support
    ts_gen = TsGenerator(settings={"undistort": True, "fisheye_lens": True, "crop_to_idxs": [10, 100, 10, 100], "new_camera_matrix": torch.eye(3)})
    assert ts_gen.fisheye_lens_used
    ts_gen = TsGenerator(settings={"undistort": True, "fisheye_lens": False})
    assert not ts_gen.fisheye_lens_used
    ts_gen = TsGenerator(settings={})
    assert not (ts_gen.settings["undistort"] and ts_gen.fisheye_lens_used)

    # Test functionality
    settings = {"shape": [184, 240], "delta_t": 0.01}
    num_events = 1000
    ts_gen = TsGenerator(settings=settings)
    fix_seed()
    x = torch.randint(settings["shape"][0], size=[num_events])
    y = torch.randint(settings["shape"][1], size=[num_events])
    t = torch.sort(torch.rand(num_events))[0] * 0.1
    p = torch.randint(2, size=[num_events])

    # Feed events
    for i in range(num_events):
        ts_gen.update(t[i], x[i], y[i], p[i])

    # Get time surface and check sanity
    ts = ts_gen.get_ts()
    assert torch.allclose(torch.max(ts), torch.tensor([1.]))
    assert torch.allclose(torch.min(ts), torch.tensor([0.]))
    assert not torch.allclose(torch.sum(ts, axis=2), torch.tensor([0.]))
    assert ts.shape, [184, 240, 2]

    # Test with multiple delta_t cannels
    settings["delta_t"] = [0.005, 0.01, 0.02, 0.04, 0.08]
    ts_gen_multi = TsGenerator(settings=settings)
    ts_gen_multi.time_stamps = ts_gen.time_stamps  # Skip event feeding

    # Get time surface and check sanity
    ts_multi = ts_gen_multi.get_ts()
    assert torch.allclose(torch.max(ts_multi), torch.tensor([1.]))
    assert torch.allclose(torch.min(ts_multi), torch.tensor([0.]))
    assert not torch.allclose(torch.sum(ts_multi, axis=2), torch.tensor([0.]))
    assert ts_multi.shape, [184, 240, 10]
    assert torch.allclose(ts_multi[:,:,1], ts[:,:,0])  # neg events dt=0.01
    assert torch.allclose(ts_multi[:,:,6], ts[:,:,1])  # pos events dt=0.01

def measure_timing():
    num_events = 1000000
    ts_gen = TsGenerator()
    fix_seed()
    x = torch.randint(ts_gen.settings["shape"][0], size=[num_events])
    y = torch.randint(ts_gen.settings["shape"][1], size=[num_events])
    t = torch.sort(torch.rand(num_events))[0]
    p = torch.randint(2, size=[num_events])

    start_time = time()
    for i in range(num_events):
        ts_gen.update(t[i], x[i], y[i], p[i])
    torch.cuda.synchronize()
    time_elapsed = time() - start_time
    print("TsGenerator.update(): Time per event:", time_elapsed / num_events)

    num_surfaces = 10000
    start_time = time()
    for i in range(num_surfaces):
        ts_gen.get_ts()
    torch.cuda.synchronize()
    time_elapsed = time() - start_time
    print("TsGenerator.get_ts(): Time per generated time surface (2 channels):", time_elapsed / num_surfaces)

    num_surfaces = 10000
    ts_gen.channels_dt = torch.tensor([0.005, 0.01, 0.02, 0.04, 0.08])
    start_time = time()
    for i in range(num_surfaces):
        ts_gen.get_ts()
    torch.cuda.synchronize()
    time_elapsed = time() - start_time
    print("TsGenerator.get_ts(): Time per generated time surface (10 channels):", time_elapsed / num_surfaces)

test_ts()
measure_timing()