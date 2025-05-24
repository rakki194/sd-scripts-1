import os
import shutil
import random
from pathlib import Path

# Configuration
SRC_DIR = '/home/kade/datasets/abandoned/1_abandoned'
DST_DIR = '/home/kade/datasets/abandoned_subset/1_abandoned'
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.jxl'}
SUBSET_SIZE = 40


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    src_path = Path(SRC_DIR)
    dst_path = Path(DST_DIR)

    # Find all image files
    image_files = [f for f in src_path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
    if len(image_files) < SUBSET_SIZE:
        raise ValueError(f"Not enough images in source directory: found {len(image_files)}, need {SUBSET_SIZE}")

    # Randomly select images
    selected_images = random.sample(image_files, SUBSET_SIZE)
    selected_basenames = {img.stem for img in selected_images}

    print(f"Selected {len(selected_images)} images:")
    for img in selected_images:
        print(f"  {img.name}")

    # Find and copy all files with the same basename
    all_files = list(src_path.iterdir())
    copied_files = []
    for basename in selected_basenames:
        matching_files = [f for f in all_files if f.stem == basename]
        for f in matching_files:
            dst_file = dst_path / f.name
            shutil.copy2(f, dst_file)
            copied_files.append(dst_file)

    print(f"\nCopied {len(copied_files)} files to {DST_DIR}:")
    for f in copied_files:
        print(f"  {f}")

if __name__ == '__main__':
    main() 