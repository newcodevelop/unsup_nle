import kagglehub
import os
from tqdm import tqdm
import shutil

print("Downloading Flickr30k dataset...")
dataset_path = kagglehub.dataset_download("hsankesara/flickr-image-dataset")

output_dir = "flickr30k_images"
os.makedirs(output_dir, exist_ok=True)

print("Copying files to output directory with progress bar...")
all_files = [
    os.path.join(root, f)
    for root, _, files in os.walk(dataset_path)
    for f in files
]

for src_file in tqdm(all_files, desc="Copying"):
    rel_path = os.path.relpath(src_file, dataset_path)
    dst_file = os.path.join(output_dir, rel_path)
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    shutil.copy2(src_file, dst_file)

print(f"Dataset copied to {output_dir}")
