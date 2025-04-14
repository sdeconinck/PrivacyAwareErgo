#!/bin/bash

# Check if two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <output_dataset_folder> <input_videos_folder>"
    exit 1
fi

# Assigning arguments to variables
OUTPUT_FOLDER=$1
INPUT_FOLDER=$2

# Check if the input folder exists
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Error: Input folder '$INPUT_FOLDER' does not exist."
    exit 1
fi

# Create the output folder if it doesn't exist
if [ ! -d "$OUTPUT_FOLDER" ]; then
    mkdir -p "$OUTPUT_FOLDER"
    echo "Created output dataset folder: $OUTPUT_FOLDER"
fi

# within the output folder, make subfolders images and labels
mkdir -p "$OUTPUT_FOLDER/images"
mkdir -p "$OUTPUT_FOLDER/labels"

# make subfolder train val test within images and labels
mkdir -p "$OUTPUT_FOLDER/images/train"
mkdir -p "$OUTPUT_FOLDER/images/val"
mkdir -p "$OUTPUT_FOLDER/images/test"
mkdir -p "$OUTPUT_FOLDER/labels/train"
mkdir -p "$OUTPUT_FOLDER/labels/val"
mkdir -p "$OUTPUT_FOLDER/labels/test"

# loop over all folders in input_folder/train
for folder in "$INPUT_FOLDER/train"/*; do
    # get the folder name
    # use script videos_to_frames.sh to convert videos to frames
    folder_name=$(basename "$folder")
    source scripts/videos_to_frames.sh "$folder" "$OUTPUT_FOLDER/images/train"
done

# loop over all folders in input_folder/val
for folder in "$INPUT_FOLDER/val"/*; do
    # get the folder name
    # use script videos_to_frames.sh to convert videos to frames
    folder_name=$(basename "$folder")
    source scripts/videos_to_frames.sh "$folder" "$OUTPUT_FOLDER/images/val"
done

# loop over all folders in input_folder/test
for folder in "$INPUT_FOLDER/test"/*; do
    # get the folder name
    # use script videos_to_frames.sh to convert videos to frames
    folder_name=$(basename "$folder")
    source scripts/videos_to_frames.sh "$folder" "$OUTPUT_FOLDER/images/test"
done

# for all frames in output_folder/images/train, create corresponding label files in output_folder/labels/train
python3 scripts/save_keypoints.py "$OUTPUT_FOLDER/images/train"
# for all frames in output_folder/images/val, create corresponding label files in output_folder/labels/val
python3 scripts/save_keypoints.py "$OUTPUT_FOLDER/images/val"
# for all frames in output_folder/images/test, create corresponding label files in output_folder/labels/test
python3 scripts/save_keypoints.py "$OUTPUT_FOLDER/images/test"
