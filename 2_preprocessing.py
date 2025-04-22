import pillow_heif
pillow_heif.register_heif_opener()
import os
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
torch.set_num_threads(2)
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights

# Constants
IMAGE_FOLDER = "images_only"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".heic", ".heif", ".dng")

weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
deeplab_model = deeplabv3_mobilenet_v3_large(weights=weights).eval()

# Resize and normalize for CLIP
resize_transform = Compose([
    Resize(224, interpolation=Image.BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073),
              (0.26862954, 0.26130258, 0.27577711))
])

def preprocess_image(fname):
    try:
        img_path = os.path.join(IMAGE_FOLDER, fname)
        img = Image.open(img_path).convert("RGB")
        img = img.resize((384, 384))
        img_tensor = TF.to_tensor(img).unsqueeze(0)

        with torch.no_grad():
            output = deeplab_model(img_tensor)['out'][0]
            mask = output.argmax(0) != 0  # foreground mask
        mask = mask.float().unsqueeze(0)
        background_only = img_tensor * (1 - mask)
        neutral_background = background_only + mask * 0.5  # light gray foreground
        resized_img = TF.to_pil_image(neutral_background.squeeze(0).clamp(0, 1))
        resized = resize_transform(resized_img)
        return resized, fname
    except Exception as e:
        print(f"Failed to process {fname}: {e}")
        return None

def load_preprocessed_images():
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(IMAGE_EXTS)]
    print(f"Found {len(image_files)} images.")
    results = [preprocess_image(f) for f in tqdm(image_files)]
    processed = [r for r in results if r is not None]
    tensors, filenames = zip(*processed)
    return tensors, filenames

if __name__ == "__main__":
    tensors, filenames = load_preprocessed_images()
    __all__ = ['tensors', 'filenames']