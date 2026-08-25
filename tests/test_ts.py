from time import time
import torch

from util.eval_utils import fix_seed
from ts_generation.ts_generator import TsGenerator, TsGeneratorType

def test_ts():
    for ts_gen_type in TsGeneratorType:
        print(f"\nRunning tests for {ts_gen_type}.")

        # Test incorrect input
        if ts_gen_type == TsGeneratorType.EventCount:
            settings={"shape": [184], "events_per_px": -0.01}
        elif ts_gen_type == TsGeneratorType.TimeWindow:
            settings={"shape": [184], "delta_t": -0.01}
        else:
            raise NotImplementedError("No test implemented for TsGenType {ts_gen_type}.")
        ts_gen = TsGenerator(tsGenType=ts_gen_type, settings=settings)
        assert len(ts_gen.settings["shape"]) == 2
        if ts_gen_type == TsGeneratorType.TimeWindow:
            assert torch.all(ts_gen.ts_dim > 0)

        # Test functionality
        settings = {"shape": [184, 240]}
        num_events = 5000
        ts_gen = TsGenerator(tsGenType=ts_gen_type, settings=settings)
        fix_seed()
        x = torch.randint(settings["shape"][0], size=[num_events])
        y = torch.randint(settings["shape"][1], size=[num_events])
        t = torch.sort(torch.rand(num_events))[0] * 0.2
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

        # Test with multiple cannels
        if ts_gen_type == TsGeneratorType.EventCount:
            # Test with multiple channels
            settings["events_per_px"] = [0.01, 0.03, 0.1, 0.3, 1.0]
            ts_gen_multi = TsGenerator(tsGenType=ts_gen_type, settings=settings)
            ts_gen_multi.buffer_events_t = ts_gen.buffer_events_t  # Skip event feeding
            ts_gen_multi.buffer_events_x = ts_gen.buffer_events_x
            ts_gen_multi.buffer_events_y = ts_gen.buffer_events_y
            ts_gen_multi.buffer_events_p = ts_gen.buffer_events_p
            print("The following warning is intended.")
        elif ts_gen_type ==  TsGeneratorType.TimeWindow:
            settings["delta_t"] = [0.005, 0.01, 0.02, 0.05, 0.1]
            ts_gen_multi = TsGenerator(tsGenType=ts_gen_type, settings=settings)
            ts_gen_multi.time_stamps = ts_gen.time_stamps  # Skip event feeding

        # Get time surface and check sanity
        ts_multi = ts_gen_multi.get_ts()
        assert torch.allclose(torch.max(ts_multi), torch.tensor([1.]))
        assert torch.allclose(torch.min(ts_multi), torch.tensor([0.]))
        assert not torch.allclose(torch.sum(ts_multi, axis=2), torch.tensor([0.]))
        assert ts_multi.shape, [184, 240, 10]

        # TS order is different for the ts gen types
        if ts_gen_type == TsGeneratorType.EventCount:
            assert torch.allclose(ts_multi[:,:,4], ts[:,:,0])  # channel 0 and 1 are like default setting
            assert torch.allclose(ts_multi[:,:,5], ts[:,:,1])
        elif ts_gen_type ==  TsGeneratorType.TimeWindow:
            assert torch.allclose(ts_multi[:,:,4], ts[:,:,0])  # neg events dt=0.01
            assert torch.allclose(ts_multi[:,:,9], ts[:,:,1])  # pos events dt=0.01

def measure_timing():
    for ts_gen_type in TsGeneratorType:
        print(f"\nMeasuring timings for {ts_gen_type}.")

        num_events = 100000
        ts_gen = TsGenerator(tsGenType=ts_gen_type)
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

        num_surfaces = 1000
        start_time = time()
        for i in range(num_surfaces):
            ts_gen.get_ts()
        torch.cuda.synchronize()
        time_elapsed = time() - start_time
        print("TsGenerator.get_ts(): Time per generated time surface (2 channels):", time_elapsed / num_surfaces)

        num_surfaces = 1000
        if ts_gen_type == TsGeneratorType.EventCount:
            ts_gen.channels_dt = torch.tensor([0.005, 0.01, 0.02, 0.04, 0.08])
        elif ts_gen_type == TsGeneratorType.TimeWindow:
            ts_gen.num_events_per_channel = [1000, 3000, 10000, 30000, 100000]
        start_time = time()
        for i in range(num_surfaces):
            ts_gen.get_ts()
        torch.cuda.synchronize()
        time_elapsed = time() - start_time
        print("TsGenerator.get_ts(): Time per generated time surface (10 channels):", time_elapsed / num_surfaces)

test_ts()
measure_timing()