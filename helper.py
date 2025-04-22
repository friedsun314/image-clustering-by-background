import numpy as np
import os
import shutil

# Config
IMAGE_FOLDER = "images_only"
OUTPUT_FOLDER = "clustered_images"

# Flatten images_only by moving all images from any subfolders up to images_only
for root, _, files in os.walk(IMAGE_FOLDER, topdown=False):
    for fname in files:
        src = os.path.join(root, fname)
        dst = os.path.join(IMAGE_FOLDER, fname)
        if src != dst and not os.path.exists(dst):
            shutil.move(src, dst)
    # Remove empty subfolders
    if root != IMAGE_FOLDER and not os.listdir(root):
        os.rmdir(root)


# --- Select best images from similar groups ---
from PIL import Image

SIMILAR_FOLDER = os.path.join(OUTPUT_FOLDER, "similar_groups")
UNIQUE_FOLDER = os.path.join(OUTPUT_FOLDER, "unique")
REST_FOLDER = os.path.join(OUTPUT_FOLDER, "rest")

os.makedirs(UNIQUE_FOLDER, exist_ok=True)
os.makedirs(REST_FOLDER, exist_ok=True)

def image_resolution(path):
    try:
        with Image.open(path) as img:
            return img.size[0] * img.size[1]
    except:
        return 0

if os.path.exists(SIMILAR_FOLDER):
    for group_dir in os.listdir(SIMILAR_FOLDER):
        group_path = os.path.join(SIMILAR_FOLDER, group_dir)
        if os.path.isdir(group_path):
            files = [f for f in os.listdir(group_path) if os.path.isfile(os.path.join(group_path, f))]
            if not files:
                continue
            files_with_paths = [os.path.join(group_path, f) for f in files]
            # Select the image with the highest resolution
            best_image = max(files_with_paths, key=image_resolution)
            best_name = os.path.basename(best_image)
            shutil.move(best_image, os.path.join(UNIQUE_FOLDER, best_name))
            # Move the rest to 'rest'
            for path in files_with_paths:
                if path != best_image:
                    shutil.move(path, os.path.join(REST_FOLDER, os.path.basename(path)))
            # Optionally remove the empty group folder
            if not os.listdir(group_path):
                os.rmdir(group_path)
