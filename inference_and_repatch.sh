#!/bin/bash

# Ask the user for input
echo "For synthetic --> real: press 1"
echo "For real --> synthetic: press 2"
read -p "Enter your choice: " choice

# Ask the user for the folder name
read -p "Enter the name of the trained output folder that you want to use for inference (e.g.: weeds): " folderName
read -p "Enter the name of the folder that you want to do the inference on (e.g.: test_A, train_A, ...): " subfolderName
# Set default values
input_dir="data/$folderName/$subfolderName"
prompt="real $folderName"
direction="a2b"

# Adjust values based on user input
if [ "$choice" -eq 2 ]; then
    prompt="synthetic $folderName"
    direction="b2a"
fi

# Ask the user if they want to use the last model checkpoint
read -p "Do you want to use the last model checkpoint? (y/n): " use_last_checkpoint

# Set the model path based on user input
if [ "$use_last_checkpoint" == "y" ] || [ "$use_last_checkpoint" == "yes" ]; then
    model_path="output/cyclegan_turbo/$folderName/checkpoints/model_25001.pkl"
else
    read -p "Enter the checkpoint number you want to use (e.g.: 00007): " model_number
    model_path="output/cyclegan_turbo/$folderName/checkpoints/model_${model_number}.pkl"
fi

# Run the Python script with the specified arguments
python3 src/inference_unpaired.py --model_path "$model_path" \
    --input_dir "$input_dir" \
    --prompt "$prompt" --direction "$direction" \
    --output_dir "outputs/$folderName" --image_prep "no_resize" \
    --use_fp16

# Run the repatch.py script with the required inputs
python3 ./repatch.py --direction "$direction" --folderName "$folderName"