#!/bin/bash
set -e

model_name="super_lite_weights"
declare -a shapes=("176 240" "176 320" "240 320" "240 424" "256 344" "360 480" "360 640" "480 640" "720 1280")

mkdir -p onnx_models/${model_name}

for shape in "${shapes[@]}"
do
    python export/export_model.py saved_models/${model_name}.pth --config config/super_lite.yaml --out_dir onnx_models/${model_name} --dtype float16 --input_shape $shape --batch_size 2 --use_onnx
done