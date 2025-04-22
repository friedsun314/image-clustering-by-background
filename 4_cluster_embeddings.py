import time
from sklearn.cluster import MiniBatchKMeans as KMeans
import numpy as np
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import os
import shutil

IMAGE_FOLDER = "images_only"
OUTPUT_FOLDER = "clustered_images"

# Load data
embeddings = np.load("clip_image_embeddings.npy")
image_filenames = np.load("clip_filenames.npy", allow_pickle=True)

NUM_CLUSTERS = 50
print(f"Using {NUM_CLUSTERS} clusters as specified.")
# kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
# labels = kmeans.fit_predict(embeddings)


# Recursively move all images from any depth in clustered_images to images_only
if os.path.exists(OUTPUT_FOLDER):
    for root, _, files in os.walk(OUTPUT_FOLDER, topdown=False):
        for fname in files:
            src = os.path.join(root, fname)
            dst = os.path.join(IMAGE_FOLDER, fname)
            try:
                if not os.path.exists(dst):  # don't overwrite
                    shutil.move(src, dst)
                else:
                    print(f"Skipping {fname}, already exists in images_only/")
            except Exception as e:
                print(f"Failed to move {fname} back to images_only/: {e}")

    # Remove any empty folders in clustered_images
    for root, dirs, _ in os.walk(OUTPUT_FOLDER, topdown=False):
        for d in dirs:
            dir_path = os.path.join(root, d)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)

# Clean output folder
if os.path.exists(OUTPUT_FOLDER):
    for cluster_dir in os.listdir(OUTPUT_FOLDER):
        cluster_path = os.path.join(OUTPUT_FOLDER, cluster_dir)
        if os.path.isdir(cluster_path):
            shutil.rmtree(cluster_path)
if os.path.exists(OUTPUT_FOLDER) and not any(os.scandir(OUTPUT_FOLDER)):
    shutil.rmtree(OUTPUT_FOLDER)
os.makedirs(OUTPUT_FOLDER)

print(f"Using {NUM_CLUSTERS} clusters as specified.")
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
labels = kmeans.fit_predict(embeddings)

# Create cluster folders
for i in range(NUM_CLUSTERS):
    os.makedirs(os.path.join(OUTPUT_FOLDER, f"cluster_{i}"), exist_ok=True)

# Move images to clusters
for fname, label in tqdm(zip(image_filenames, labels), total=len(labels), desc="Clustering images"):
    src = os.path.join(IMAGE_FOLDER, fname)
    dst = os.path.join(OUTPUT_FOLDER, f"cluster_{label}", fname)
    if not os.path.exists(src):
        print(f"Warning: {src} not found. Skipping.")
        continue
    shutil.move(src, dst)

print("Done clustering.")