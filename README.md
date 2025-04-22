# Background Image Sorting

A Python pipeline for grouping and clustering images based on **visual background similarity**, using CLIP embeddings and KMeans clustering.

This project helps organize large batches of photos, especially when metadata is missing, by clustering visually similar backgrounds.

---

## 🔧 Pipeline Overview

1. **`1_separate.py`**  
   Extracts and sorts input files (e.g., into image/video folders).

2. **`2_preprocessing.py`**  
   - Resizes images  
   - Applies background segmentation using DeepLabV3  
   - Converts images into tensors ready for CLIP

3. **`3_batch_encode.py`**  
   - Loads the CLIP model  
   - Encodes all background-focused images into vector embeddings  
   - Saves `clip_image_embeddings.npy`

4. **`4_cluster_embeddings.py`**  
   - Loads embeddings and clusters them using MiniBatchKMeans  
   - Automatically flattens folders, restores original images if re-run  
   - Outputs a `clustered_images/` folder containing organized clusters

5. **`helper.py`**  
   Contains utility functions used across the project (if applicable).

---

## 📁 File Structure

```
.
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── 1_separate.py
├── 2_preprocessing.py
├── 3_batch_encode.py
├── 4_cluster_embeddings.py
├── helper.py
```

---

## 📥 Installation

```bash
git clone https://github.com/yourusername/background-image-sorting.git
cd background-image-sorting
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Usage

Run the pipeline step-by-step (recommended in order):

```bash
python 1_separate.py
python 2_preprocessing.py
python 3_batch_encode.py
python 4_cluster_embeddings.py
```

---

## 🧠 Technologies Used

- [CLIP](https://github.com/openai/CLIP) — for image embeddings
- [TorchVision](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_mobilenet_v3_large.html) — for DeepLabV3 background segmentation
- `sklearn`, `tqdm`, `Pillow`, `numpy`, `shutil`, `os`

---

## 📃 License

MIT License

---

## 🙋‍♂️ Contributing

Feel free to open issues or submit pull requests!
