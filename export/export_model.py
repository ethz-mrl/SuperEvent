import argparse
import numpy as np
import os
import yaml

import torch

from models.super_event import SuperEvent, SuperEventFullRes

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("model", default="", help="Model weights to be traced")
parser.add_argument("--config", default="config/super_event.yaml", help="Parameter configuration.")
parser.add_argument("--out_dir", default="saved_models", help="Directory where traced model will be saved.")
parser.add_argument("--dynamic_shapes", action=argparse.BooleanOptionalAction, help="Support dynamic shapes.")
parser.add_argument("--input_shape",  nargs="*", default=[480, 640], type=int, help="Pixel resolution of input (H, W).")
parser.add_argument("--batch_size",  default=1, type=int, help="Batch size if greater 1.")
parser.add_argument("--dtype",  default="float32", type=str, help="Datatype to be used (float16 or half will convert model to half precision)")
parser.add_argument("--use_onnx", action=argparse.BooleanOptionalAction, help="Optimize with TensorRT.")
args = parser.parse_args()

assert args.batch_size > 0

if args.use_onnx:
    import onnxruntime

# Define model name
if args.use_onnx:
    prefix = "onnx"
    ext = "onnx"
    if args.dynamic_shapes:
        postfix = "dyn_shapes"
    else:
        postfix = f"{args.input_shape[0]}x{args.input_shape[1]}"
    postfix += f"_batch_size_{args.batch_size}"
else:
    prefix = "traced"
    postfix = "dyn_shapes"
    ext = "pt2"
out_path = os.path.join(args.out_dir, f"{prefix}_{os.path.splitext(os.path.basename(args.model))[0]}_{postfix}_{args.dtype}.{ext}")

# Load config
with open(args.config, 'r') as f:
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
print(f"Using config:\n\n{yaml.dump(config)}")

# Load model
if config["pixel_wise_predictions"]:
    model = SuperEventFullRes(config, tracing=True)
else:
    model = SuperEvent(config, tracing=True)
model.load_state_dict(torch.load(args.model, weights_only=True), strict=True)
model.eval()
print("Loaded model weights from", args.model)

device = (
    "cuda"
    if args.use_onnx
    else "cpu"
)
model.to(device)

# An example input you would normally provide to your model's forward() method.
example = torch.rand(args.batch_size, config["input_channels"], args.input_shape[0], args.input_shape[1]).to(device)

if args.dtype == "float16" or args.dtype == "half":
    model.to(torch.float16)
    example = example.to(torch.float16)
    print("Converted model to half precision (float16)")
elif not args.use_onnx:
    torch.set_float32_matmul_precision('high')

#ret = model.forward(torch.ones((2, config["input_channels"], 160, 240), device=device)) # For debugging

# if args.use_onnx:
#     min_shape_possible = config["grid_size"]
#     if "backbone_config" in config:
#         min_shape_possible = 2 ** (len(config["backbone_config"]["num_blocks"]) - 1) * \
#                                     config["backbone_config"]["stem"]["patch_size"] * \
#                                     np.max(config["backbone_config"]["stage"]["attention"]["partition_size"])
#         min_shape_possible = min_shape_possible.item()

#     inputs = [
#         torch_tensorrt.Input(
#             min_shape=[1, config["input_channels"], min_shape_possible, min_shape_possible],
#             opt_shape=[args.batch_size, config["input_channels"], args.input_shape[0], args.input_shape[1]],
#             max_shape=[args.batch_size, config["input_channels"], args.input_shape[0], args.input_shape[1]],
#             dtype=torch.half,
#         )
#     ]
#     enabled_precisions = {torch.float, torch.half}  # Run with fp16

#     trt_ts_module = torch_tensorrt.compile(
#         model, inputs=inputs, ir="ts", enabled_precisions=enabled_precisions
#     )

#     # Save model
#     torch.jit.save(trt_ts_module, out_path)

# else:
with torch.no_grad():

    if args.use_onnx:
        dynamic_axes=None
        if args.dynamic_shapes:
            dynamic_axes={"mcts": {2: "height", 3: "width"},
                          "keypoint_map": {1: "height", 2: "width"},
                          "descriptor_grid": {2: "height", 3: "width"},
                         }

        # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
        # Warning is caused by asserts in maxvit.py
        print("Converting model to onnx...")
        onnx_program = torch.onnx.export(model,
                                        example,
                                        dynamo=True,
                                        input_names=["mcts"],
                                        output_names=["keypoint_map", "descriptor_grid"],
                                        dynamic_axes=dynamic_axes
                                        )
        onnx_program.optimize()
        out_path = f"{out_path}"
        onnx_program.save(out_path)

    else:
        script_module = torch.jit.trace(model, example)
        script_module = torch.jit.freeze(script_module)

        # Save model
        script_module.save(out_path)

print(f"Model saved as {out_path}")
