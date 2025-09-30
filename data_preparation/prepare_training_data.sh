#!/bin/bash
set -e

function get_split_dir(){
    if [[ -d $1/train/$2/frames ]]; then
        split_dir="train"
    elif [[ -d $1/val/$2/frames ]]; then
        split_dir="val"
    elif [[ -d $1/test/$2/frames ]]; then
        split_dir="test"
    elif [[ -d $1/$2/frames ]]; then
        split_dir=""
    else
        echo "Sequence $2 not found."
        exit 1
    fi
    echo $split_dir
}

# Fix python imports
export PYTHONPATH=$PYTHONPATH:$pwd

# All data will be loaded from a source dir in the downloaded dataset 
# and saved in a target dir with the required format for training

# Set time surface delta
delta_t="0.001 0.003 0.01 0.03 0.1"

# Evaluate flags
while getopts d:t:c:u flag
do
    case "${flag}" in
        d) data_dir=${OPTARG};;
        t) target_dir=${OPTARG};;
        c) config_path=${OPTARG};;
        u) undistort=True;;
    esac
done

# Check inputs
if [[ -z ${data_dir} ]]; then
    echo "Please specify the directory of the training dataset with the flag '-d'."
    exit 1
fi

if [[ -z ${target_dir} ]]; then
    target_dir=${data_dir}/../SuperEvent_data
    mkdir -p ${target_dir}
    echo "Created target directory at path ${target_dir}. You can use the flag '-t' to specify the target path."
fi

if [[ -z ${config_path} ]]; then
    config_path=${HOME}/repos/SuperEvent/config/super_event.yaml
    if [[ ! -e ${config_path} ]]; then
        echo "Please specify the path of the config file with the flag '-c'."
        exit 1
    fi
    echo "Assuming config file path as ${config_path}. You can use the flag '-c' to adjust the config file path."
fi

if [ "$undistort" = "" ]; then
    echo "Undistortion of images and time surfaces is turned off by default. You can turn it on with the flag '-u'."
else
    echo "Undistortion turned on. Assuming a 'calib.txt' file in every sequence directory."
fi

# Get dataset name
dataset_name=${data_dir%*/}      # remove the trailing "/"
dataset_name="${dataset_name##*/}"    # print everything after the final "/"

# Step trough all sequences in dataset
for dir in ${data_dir}/*/     # list directories in the form "/tmp/dirname/"
do
    dir=${dir%*/}      # remove the trailing "/"
    seq="${dir##*/}"    # print everything after the final "/"
    echo -e "\nProcessing sequence ${dir}."

    sequence_target_dir=${target_dir}/${dataset_name}/${seq}
    mkdir -p ${sequence_target_dir}

    # Step 1: Convert images
    python data_preparation/convert_images.py ${data_dir}/${seq} ${sequence_target_dir} ${dataset_name} --delta_t ${delta_t} --undistort "$undistort"

    # Step 2: Generate time surfaces
    python data_preparation/events2ts.py ${data_dir}/${seq} ${sequence_target_dir}/time_surfaces ${dataset_name} --timestamp_path ${sequence_target_dir}/frame_timestamps.txt --delta_t ${delta_t} --undistort "$undistort"

    # Step 3: Detect and match features with SuperGlue
    if [[ -d ${sequence_target_dir}/sg_matches ]] || [[ -d ${target_dir}/train/${dataset_name}/${seq}/sg_matches ]]  || [[ -d ${target_dir}/val/${dataset_name}/${seq}/sg_matches ]]  || [[ -d ${target_dir}/test/${dataset_name}/${seq}/sg_matches ]]; then
        echo "Directory ${target_dir}/</;train/;val/;test/>${dataset_name}/${seq}/sg_matches already exists and will not be overwritten. Please delete it manually to recompute SuperGlue matches."
    else
        split_dir=$(get_split_dir ${target_dir} ${dataset_name}/${seq})
        python data_preparation/prepare_pseudo_groundtruth.py ${target_dir}/${split_dir}/${dataset_name}/${seq}
    fi
done

# Step 6: Split up sequences into train, val and test
python data_preparation/split_data.py ${target_dir} ${config_path}