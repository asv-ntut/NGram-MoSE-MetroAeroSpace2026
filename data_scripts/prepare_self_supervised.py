import os
import glob
import cv2
import numpy as np
import rasterio
from tqdm import tqdm

# ================= Configuration =================
SOURCE_DIR = "dataset_source_raw"
TARGET_DIR = "../dataset_self_supervised" 

SCALE = 3
HR_TILE_SIZE = 480 
LR_TILE_SIZE = HR_TILE_SIZE // SCALE

VAL_SPLIT_RATIO = 0.1
TEST_SPLIT_RATIO = 0.1

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_scene(tif_path, split_dirs):
    print(f"Processing: {os.path.basename(tif_path)}")
    
    with rasterio.open(tif_path) as src:
        img = src.read().transpose(1, 2, 0) # (H, W, C)
    
    h, w, c = img.shape
    n_h = h // HR_TILE_SIZE
    n_w = w // HR_TILE_SIZE
    
    print(f" -> Dimensions: {h}x{w}. Tiles: {n_h*n_w}")
    
    filename_base = os.path.splitext(os.path.basename(tif_path))[0]
    
    count = 0
    for i in range(n_h):
        for j in range(n_w):
            y = i * HR_TILE_SIZE
            x = j * HR_TILE_SIZE
            hr_crop = img[y:y+HR_TILE_SIZE, x:x+HR_TILE_SIZE, :]
            
            # Generate LR (Synthetic)
            img_blurred = cv2.GaussianBlur(hr_crop, (0, 0), sigmaX=1.0, sigmaY=1.0)
            lr_crop = cv2.resize(img_blurred, (LR_TILE_SIZE, LR_TILE_SIZE), interpolation=cv2.INTER_CUBIC)
            
            # Split
            rand = np.random.rand()
            if rand < TEST_SPLIT_RATIO:
                subset = "test"
            elif rand < (TEST_SPLIT_RATIO + VAL_SPLIT_RATIO):
                subset = "val"
            else:
                subset = "train"
                
            # Save
            name = f"{filename_base}_tile_{i}_{j}.tif"
            
            with rasterio.open(
                os.path.join(split_dirs[f"{subset}_hr"], name), 'w',
                driver='GTiff', height=HR_TILE_SIZE, width=HR_TILE_SIZE,
                count=3, dtype=hr_crop.dtype
            ) as dst:
                dst.write(hr_crop.transpose(2, 0, 1))
                
            with rasterio.open(
                os.path.join(split_dirs[f"{subset}_lr"], name), 'w',
                driver='GTiff', height=LR_TILE_SIZE, width=LR_TILE_SIZE,
                count=3, dtype=lr_crop.dtype
            ) as dst:
                dst.write(lr_crop.transpose(2, 0, 1))
            
            count += 1
            
    print(f" -> Generated {count} pairs.")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    # Fix: download_train_data.py saves to CWD/dataset_source_raw (Project Root)
    # So we should look for it in root_dir, not current_dir
    src_dir_full = os.path.join(root_dir, SOURCE_DIR)
    target_dir_full = os.path.join(root_dir, "dataset_self_supervised")
    
    if not os.path.exists(src_dir_full):
        print(f"Source directory not found: {src_dir_full}")
        return

    subsets = ["train", "test", "val"]
    split_dirs = {}
    
    for s in subsets:
        hr = os.path.join(target_dir_full, f"{s}_hr")
        lr = os.path.join(target_dir_full, f"{s}_lr")
        ensure_dir(hr)
        ensure_dir(lr)
        split_dirs[f"{s}_hr"] = hr
        split_dirs[f"{s}_lr"] = lr
        
    tif_files = glob.glob(os.path.join(src_dir_full, "*.tif"))
    
    print(f"Found {len(tif_files)} scenes. Starting slicing...")
    
    for tif in tqdm(tif_files):
        process_scene(tif, split_dirs)
        
    print("\nDataset Preparation Complete!")

if __name__ == "__main__":
    main()
