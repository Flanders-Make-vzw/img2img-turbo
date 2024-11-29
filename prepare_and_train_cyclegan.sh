#!/bin/bash

echo "What is the name of the folder located in /data that you want to use for training? (without quotation marks)"
read folderName
read -p "Enter the number of epochs: " num_epochs
# Change directory to the specified folder
cd "data/$folderName/" 

# Rename folders
mv "synthetic" "train_A_original"
mv "real" "train_B_original"

# Create test folders
mkdir -p "test_A_original"
mkdir -p "test_B_original"

# Function to move 2% of images to test folder
move_images() {
    local sourceFolder="$1"
    local destFolder="$2"

    # Count total images
    totalImages=$(find "$sourceFolder" -type f | wc -l)

    # Calculate number of images to move (2%)
    numImagesToMove=$(( (totalImages + 49) / 50 ))

    # Randomly select and move images
    find "$sourceFolder" -type f | shuf -n "$numImagesToMove" | while read -r file; do
        mv "$file" "$destFolder"
    done
}

move_images "train_A_original" "test_A_original"
move_images "train_B_original" "test_B_original"

# Create fixed_prompt_A.txt with the words "synthetic <folderName>"
echo "synthetic $folderName" > fixed_prompt_a.txt

# Create fixed_prompt_B.txt with the words "real <folderName>"
echo "real $folderName" > fixed_prompt_b.txt

# Change directory to the specified folder
cd "../.." 
# Run the patches Python script
echo "Running patches.py..."
python3 patches.py "$folderName"
if [ $? -ne 0 ]; then
    echo "Failed to run patches.py"
    exit 1
fi

# Run training script
accelerate launch --main_process_port 29501 src/train_cyclegan_turbo.py \
    --pretrained_model_name_or_path="stabilityai/sd-turbo" \
    --output_dir="output/cyclegan_turbo/$folderName" \
    --dataset_folder "data/$folderName" \
    --train_img_prep "no_resize" --val_img_prep "no_resize" \
    --learning_rate="1e-5" --max_train_steps=$num_epochs \
    --train_batch_size=1 --gradient_accumulation_steps=1 \
    --tracker_project_name "gparmar_unpaired_h2z_cycle_debug_v2" \
    --enable_xformers_memory_efficient_attention --validation_steps 14501 \
    --lambda_gan 0.5 --lambda_idt 1 --lambda_cycle 1 --no-logging