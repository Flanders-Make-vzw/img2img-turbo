#!/bin/bash

echo "What is the name of the folder located in /data that you want to use for training? (without quotation marks)"
read folderName

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
