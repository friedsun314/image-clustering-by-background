import torch
import clip
import numpy as np
from tqdm import tqdm
from preprocessing import load_preprocessed_images
import time

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    model, _ = clip.load("ViT-B/32", device=DEVICE)
    model.eval()
    tensors, filenames = load_preprocessed_images()

    assert len(tensors) == len(filenames), "Mismatch between tensors and filenames!"

    embeddings = []
    start = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, len(tensors), BATCH_SIZE)):
            batch = torch.stack(tensors[i:i + BATCH_SIZE]).to(DEVICE)
            batch_emb = model.encode_image(batch).cpu().numpy()
            embeddings.append(batch_emb)

    all_embeddings = np.concatenate(embeddings, axis=0)
    print(f"Embeddings shape: {all_embeddings.shape}")
    np.save("clip_image_embeddings.npy", all_embeddings)
    np.save("clip_filenames.npy", filenames)  # Save filenames to preserve order
    print(f"Encoding took {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    main()