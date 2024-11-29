import os
import cv2 as cv
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import math
import argparse
from PIL import Image

def blend_horizontal(A, B, overlap):
    height, width, channels = A.shape
    overlap = int(overlap)
    padded_A = np.zeros((height, width * 2 - overlap, channels), dtype=A.dtype)
    padded_B = np.zeros((height, width * 2 - overlap, channels), dtype=B.dtype)

    padded_A[:, :width] = A
    padded_B[:, width - overlap:] = B
    mask = np.zeros((height, width * 2 - overlap, channels), dtype=np.float32)
    mask[:, :width - overlap] = 1
    gradient = np.linspace(1, 0, overlap).reshape(1, -1, 1)
    mask[:, width - overlap:width] = np.tile(gradient, (height, 1, channels))

    blended_image = (padded_A * mask + padded_B * (1 - mask)).astype(np.uint8)
    return blended_image

def blend_vertical(row_images, overlap_y, patch_size, reconstructed_image=None):
    i = -1
    for C, D in zip(deepcopy(row_images[:-1]), deepcopy(row_images[1:])):
        i += 1
        height, width, channels = row_images[1].shape
        overlap = overlap_y * patch_size
        overlap = int(overlap)
        padded_A = np.zeros((height * 2 - overlap, width, channels), dtype=C.dtype)
        padded_B = np.zeros((height * 2 - overlap, width, channels), dtype=D.dtype)

        padded_A[:height, :, :] = C
        padded_B[height - overlap:, :, :] = D
        mask = np.zeros((height * 2 - overlap, width, channels), dtype=np.float32)
        mask[:height - overlap, :] = 1
        gradient = np.linspace(1, 0, overlap).reshape(-1, 1, 1)
        mask[height - overlap:height, :] = np.tile(gradient, (1, width, channels))

        blended_image = (padded_A * mask + padded_B * (1 - mask)).astype(np.uint8)
        height, width = blended_image.shape[:2]
        crop = blended_image[int(height // 4):int(3 * height // 4), :]
        if i == 0:
            reconstructed_image[int(patch_size * i + height // 4):int(patch_size * i + 3 * height // 4), :] = crop
          
        else:
            reconstructed_image[int(patch_size * i - i * overlap + height // 4):int(patch_size * i - i * overlap + 3 * height // 4), :] = crop
    return reconstructed_image

def reconstruct_image(patched_folder, output_folder,original_width, original_height, patch_size, overlap_x, overlap_y,horizontal_patches):
    patches_by_base_name = {}

    for filename in os.listdir(patched_folder):
        if filename.endswith((".jpg", ".png")):
            base_name = os.path.splitext(filename)[0].rsplit('_', 2)[0]
            patches_by_base_name.setdefault(base_name, []).append(filename)

    for base_name, filenames in tqdm(patches_by_base_name.items(), desc="Reconstructing images"):
        reconstructed_image = np.zeros((original_height, original_width, 3), dtype=np.uint8)
        row_images = []
        patch_counter = 0

        for filename in filenames:
            patch_counter += 1
            parts = filename.split('_')
            row, col = int(parts[-2]), int(parts[-1].split('.')[0])
            y = int((row - 1) * (patch_size * (1 - overlap_y)))
            x = int((col - 1) * (patch_size * (1 - overlap_x)))

            patch_before = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
            if x != 0:
                previous_col = col - 1 if col > 1 else col
                previous_filename = f"{base_name}_{row}_{previous_col}.png"
                previous_patch_path = os.path.join(patched_folder, previous_filename)
                if os.path.exists(previous_patch_path):
                    patch_before = cv.imread(previous_patch_path)
                else:
                    previous_filename = f"{base_name}_{row}_{previous_col}.jpg"
                    previous_patch_path = os.path.join(patched_folder, previous_filename)
                    if os.path.exists(previous_patch_path):
                        patch_before = cv.imread(previous_patch_path)
                    else:
                        print(f"Previous patch not found: {previous_patch_path}")

            patch_image = cv.imread(os.path.join(patched_folder, filename))

            if x == 0:
                reconstructed_image[y:y + patch_size, x:x + patch_size] = patch_image

            else:
                blended_image = blend_horizontal(patch_before, patch_image, overlap=math.ceil(overlap_x * patch_size))
                crop = blended_image[:, int(patch_size - overlap_x * patch_size):2 * patch_size]
                reconstructed_image[y:y + patch_size, x:x + patch_size] = crop

            if x == original_width - patch_size and patch_counter == horizontal_patches:
                row_images.append(deepcopy(reconstructed_image[y:y + patch_size, :]))
                patch_counter = 0

        final_image = blend_vertical(row_images, overlap_y, patch_size, reconstructed_image)
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, base_name + '.png')
        cv.imwrite(output_path, final_image)
        
        
def calculate_overlap_sizes(width, height, patch_size):
    horizontal_patches = width / patch_size
    vertical_patches = height / patch_size

    horizontal_patches = horizontal_patches + 1 if horizontal_patches % 1 == 0 else math.ceil(horizontal_patches)
    vertical_patches = vertical_patches + 1 if vertical_patches % 1 == 0 else math.ceil(vertical_patches)

    overlap_horizontal = ((horizontal_patches * patch_size - width) / (horizontal_patches - 1)) / patch_size * 100
    overlap_vertical = ((vertical_patches * patch_size - height) / (vertical_patches - 1)) / patch_size * 100

    return overlap_horizontal / 100, overlap_vertical / 100, horizontal_patches

def get_image_dimensions(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    if not image_files:
        raise FileNotFoundError(f"No image files found in {folder_path}")
    first_image_path = os.path.join(folder_path, image_files[0])
    with Image.open(first_image_path) as img:
        return img.width, img.height

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct images from patches.")
    parser.add_argument("--direction", type=str, required=True, help="Direction of the transformation (a2b or b2a)")
    parser.add_argument("--folderName", type=str, required=True, help="FolderName")
    args = parser.parse_args()
    direction = args.direction
    folderName=args.folderName

    input_folder_name = f"outputs/{folderName}"
    output_folder_name = input_folder_name + "_reconstructed"

    if direction == 'a2b':
        folder_path = f"data/{folderName}/train_A_original"
    elif direction == 'b2a':
        folder_path = f"data/{folderName}/train_B_original"
    else:
        raise ValueError("Invalid direction. Must be 'a2b' or 'b2a'.")

    width_original_image, height_original_image = get_image_dimensions(folder_path)
    patch_size = 256

    overlap_horizontal, overlap_vertical, horiz_patches = calculate_overlap_sizes(width_original_image, height_original_image, patch_size)
    reconstruct_image(input_folder_name, output_folder_name, width_original_image, height_original_image, patch_size, overlap_horizontal, overlap_vertical, horiz_patches)

    print(f"Written to {output_folder_name}")