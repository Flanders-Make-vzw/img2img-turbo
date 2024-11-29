import os
import cv2
import numpy as np
import math
import argparse
from tqdm import tqdm

def calculate_overlap_sizes(width, height, patch_size):
    horizontal_patches = width / patch_size
    vertical_patches = height / patch_size
    horizontal_patches = horizontal_patches + 1 if horizontal_patches % 1 == 0 else math.ceil(horizontal_patches)
    vertical_patches = vertical_patches + 1 if vertical_patches % 1 == 0 else math.ceil(vertical_patches)
    print("horizontal_patches",horizontal_patches, "vertical_patches",vertical_patches)
    overlap_horizontal = ((horizontal_patches * patch_size - width) / (horizontal_patches - 1)) / patch_size * 100
    overlap_vertical = ((vertical_patches * patch_size - height) / (vertical_patches - 1)) / patch_size * 100
    return overlap_horizontal / 100, overlap_vertical / 100

def create_patched_folder(image_folder):
    patched_folder = image_folder.replace('_original', '')
    os.makedirs(patched_folder, exist_ok=True)
    return patched_folder

def get_image_files(image_folder):
    return [f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png"))]

def calculate_step_sizes(patch_size, overlap_width, overlap_height):
    step_size_width = math.ceil(patch_size  - math.ceil(overlap_width*patch_size))
    step_size_height = math.ceil(patch_size  - math.ceil(overlap_height*patch_size))
    return step_size_width, step_size_height

def save_patch(image, x, y, patch_size, patched_folder, filename, row_counter, col_counter):
    patch = image[int(y):int(y + patch_size), int(x):int(x + patch_size)]
    patch_filename = f"{filename[:-4]}_{row_counter}_{col_counter}.png"
    patch_path = os.path.join(patched_folder, patch_filename)
    cv2.imwrite(patch_path, patch)

def extract_patches(image_folder, patch_size, overlap_width, overlap_height):
    patched_folder = create_patched_folder(image_folder)
    image_files = get_image_files(image_folder)
    step_size_width, step_size_height = calculate_step_sizes(patch_size, overlap_width, overlap_height)

    for filename in tqdm(image_files, desc="Cutting images into patches"):
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        row_counter = 1
        for y in np.arange(0, height - patch_size + 1, step_size_height):
            col_counter = 1
            for x in np.arange(0, width - patch_size + 1, step_size_width):
                save_patch(image, x, y, patch_size, patched_folder, filename, row_counter, col_counter)
                col_counter += 1
            row_counter += 1



def main(folder_name):
    input_path = f"data/{folder_name}"
    patch_size = 256
    folder_A = "train_A_original"
    folder_B = "train_B_original"

    # Read the first image to get its dimensions
    sample_image_path_A = os.path.join(input_path, folder_A, os.listdir(os.path.join(input_path, folder_A))[0])
    sample_image_A = cv2.imread(sample_image_path_A)
    height_A, width_A, _ = sample_image_A.shape

    sample_image_path_B = os.path.join(input_path, folder_B, os.listdir(os.path.join(input_path, folder_B))[0])
    sample_image_B = cv2.imread(sample_image_path_B)
    height_B, width_B, _ = sample_image_B.shape

    if height_A == height_B and width_A == width_B:
        overlap_width, overlap_height = calculate_overlap_sizes(width_A, height_A, patch_size)
        for subfolder in ["train_A_original", "test_A_original", "train_B_original", "test_B_original"]:
            extract_patches(os.path.join(input_path, subfolder), patch_size, overlap_width, overlap_height)
    else:
        overlap_width_A, overlap_height_A = calculate_overlap_sizes(width_A, height_A, patch_size)
        for subfolder in ["train_A_original", "test_A_original"]:
            extract_patches(os.path.join(input_path, subfolder), patch_size, overlap_width_A, overlap_height_A)

        overlap_width_B, overlap_height_B = calculate_overlap_sizes(width_B, height_B, patch_size)
        for subfolder in ["train_B_original", "test_B_original"]:
            extract_patches(os.path.join(input_path, subfolder), patch_size, overlap_width_B, overlap_height_B)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cut images into patches.")
    parser.add_argument("folder_name", type=str, help="The name of the folder located in data")
    args = parser.parse_args()
    main(args.folder_name)