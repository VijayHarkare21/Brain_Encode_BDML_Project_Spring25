from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import os
import numpy as np
import torch

data = np.load(r"D:\Vijay\NYU\Spring_25\BDMLS\Project\dataset\ImageNetEEG\processed_eeg_signals.npy", allow_pickle=True).item()

cache_dir = r"D:\Vijay\NYU\Spring_25\BDMLS\Project\code\image_embedding_vit\vit_cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir=cache_dir, device_map='auto')
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k', cache_dir=cache_dir)

unique_synsets = list(set([i.strip().split('_')[0].strip() for i in data['images']]))

all_img_ids = {}

base_path = r"D:\Vijay\NYU\Spring_25\BDMLS\Project\dataset\ImageNetEEG\images"

count_not_found = 0

for i in data['images']:
    img_path = os.path.join(base_path, i.split('_')[0], i + ".JPEG")

    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        count_not_found += 1
        continue

    image = Image.open(img_path)

    image = image.convert('RGB')

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs[0][:, 0, :]

    print("Processing image:", i)
    print(last_hidden_states.shape) 

    all_img_ids[i] = last_hidden_states.cpu().numpy()

print("Not found images:", count_not_found)

# Save the embeddings to a file
np.save(r"D:\Vijay\NYU\Spring_25\BDMLS\Project\code\image_embedding_vit\vit_embeddings.npy", all_img_ids)