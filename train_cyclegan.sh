#!/bin/bash

echo "What is the name of the folder located in /data that you want to use for training? (without quotation marks)"
read folderName

# Run training script
accelerate launch --main_process_port 29501 src/train_cyclegan_turbo.py \
    --pretrained_model_name_or_path="stabilityai/sd-turbo" \
    --output_dir="output/cyclegan_turbo/$folderName" \
    --dataset_folder "data/$folderName" \
    --train_img_prep "no_resize" --val_img_prep "no_resize" \
    --learning_rate="1e-5" --max_train_steps=25000 \
    --train_batch_size=1 --gradient_accumulation_steps=1 \
    --tracker_project_name "gparmar_unpaired_h2z_cycle_debug_v2" \
    --enable_xformers_memory_efficient_attention --validation_steps 14501 \
    --lambda_gan 0.5 --lambda_idt 1 --lambda_cycle 1 --no-logging