import os

# Config
SOURCE_FOLDER = "/Users/guy/Desktop/Iron Swords - Lehava/Images"
IMAGE_FOLDER = "images_only"
VIDEO_FOLDER = "videos_only"

# Supported extensions
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".heic", ".heif", ".dng")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".mts")

# Create folders if they don't exist
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)

# Move files to appropriate folders
for fname in os.listdir(SOURCE_FOLDER):
    fpath = os.path.join(SOURCE_FOLDER, fname)
    if fname.lower().endswith(IMAGE_EXTS):
        os.rename(fpath, os.path.join(IMAGE_FOLDER, fname))
    elif fname.lower().endswith(VIDEO_EXTS):
        os.rename(fpath, os.path.join(VIDEO_FOLDER, fname))