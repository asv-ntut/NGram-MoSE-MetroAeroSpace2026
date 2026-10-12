import h5py
import numpy as np
import cv2
import os
import glob
import random
import shutil
from tqdm import tqdm

# ================= Configuration =================
TRAIN_IMG_DIR = 'raw_data/TrainData/img'
TRAIN_MASK_DIR = 'raw_data/TrainData/mask'
OUTPUT_ROOT = 'landslide_workdir' 
GAMMA = 1.2

DIRS = {
    # SR Input
    'raw_h5': os.path.join(OUTPUT_ROOT, 'raw_h5'),
    'raw_h5_lr': os.path.join(OUTPUT_ROOT, 'raw_h5_lr'), # 42x42 for A3
    
    # YOLO Data
    'img_128_GT': os.path.join(OUTPUT_ROOT, 'yolo_data/img_128_GT'),
    'img_128_Bic': os.path.join(OUTPUT_ROOT, 'yolo_data/img_128_Bic'),
    'img_384_Bic': os.path.join(OUTPUT_ROOT, 'yolo_data/img_384_Bic'),
    'img_42_LR': os.path.join(OUTPUT_ROOT, 'yolo_data/img_42_LR'),
    'labels': os.path.join(OUTPUT_ROOT, 'yolo_data/labels')
}

for base_dir in DIRS.values():
    for s in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_dir, s), exist_ok=True)

# ================= Core Functions (Mirrored from SR Script) =================

def to_vis_fair(sr_img, vis_max=0.15):
    """
    【絕對公平的固定拉伸】 (Logic from run_inference_l4s_fair.py)
    使用相同的 VIS_MAX 截斷亮部，相同的 Gamma 校正。
    任何因為過曝或過暗造成的特徵損失，都會直接反映在最終圖上。
    """
    # 1. 取絕對值 (處理 SR 常見的負數 ringing artifacts，雖然這裡大多是處理正則化過的值)
    img = np.abs(sr_img)
    
    # 2. 嚴格的全局閥值截斷 (不根據圖片內容改變)
    img = np.clip(img, 0, vis_max)
    
    # 3. 正規化到 0-1
    img = img / vis_max
    
    # 4. 固定的 Gamma 校正
    if GAMMA != 1.0:
        img = np.power(img, 1.0/GAMMA)
    
    return (img * 255.0).clip(0, 255).astype(np.uint8)[:, :, ::-1] # BGR

def scientific_normalize(img):
    """
    User Suggestion 2: Scientific Norm + Clip [0, 1]
    (Logic copied from user's run_inference_l4s.py load_data_and_norm)
    """
    if img.max() > 255: img = img.astype(np.float32) / 10000.0
    elif img.max() > 1.0: 
        factors = np.array([0.04452, 0.06634, 0.08246], dtype=np.float32)
        img = img.astype(np.float32) * factors
    else: img = img.astype(np.float32)
    
    # Clip Outliers
    img = np.clip(img, 0, 1)
    return img

def create_h5_from_array(save_path, img_array):
    """Save numpy array as .h5 file with 'img' key"""
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('img', data=img_array)

# ================= Process =================

def process():
    h5_files = glob.glob(os.path.join(TRAIN_IMG_DIR, '*.h5'))
    if not h5_files:
        print(f"Error: No .h5 files found in {TRAIN_IMG_DIR}")
        return

    random.seed(42)
    random.shuffle(h5_files)
    
    total = len(h5_files)
    train_end, val_end = int(total*0.8), int(total*0.9)

    print(f"Total files: {total} (Train: {train_end}, Val: {val_end-train_end}, Test: {total-val_end})")

    for idx, h5_path in enumerate(tqdm(h5_files)):
        if idx < train_end: s = 'train'
        elif idx < val_end: s = 'val'
        else: s = 'test'
        
        basename = os.path.basename(h5_path)
        save_id = f"ls_{idx:06d}" # Unified format
        
        # --- 1. Load Data ---
        with h5py.File(h5_path, 'r') as f: 
            img_data = f['img'][:] 
        
        # Handle Format: (14, H, W) vs (H, W, 14)
        if img_data.ndim == 3 and img_data.shape[0] == 14:
            # Channel First (14, 128, 128) -> Standardize to Channel First for processing
            raw_img_c_first = img_data
            h, w = img_data.shape[1], img_data.shape[2]
            # For RGB vis: [3, 2, 1] then transpose to (H, W, 3)
            rgb_raw = scientific_normalize(img_data[[3, 2, 1], :, :].transpose(1, 2, 0))
        elif img_data.ndim == 3 and img_data.shape[2] == 14:
            # Channel Last (128, 128, 14)
            h, w = img_data.shape[0], img_data.shape[1]
            # Transpose to Channel First for consistent processing below
            raw_img_c_first = img_data.transpose(2, 0, 1) 
            # For RGB vis: select [3, 2, 1] directly
            rgb_raw = scientific_normalize(img_data[:, :, [3, 2, 1]])
        else:
            print(f"Skipping {basename}: Unknown shape {img_data.shape}")
            continue

        # Save raw 128x128 .h5 (For SR B3) - Keep original shape/data
        shutil.copyfile(h5_path, os.path.join(DIRS['raw_h5'], s, f"{save_id}.h5"))

        # Create raw 42x42 .h5 (For SR A3 start point)
        # We resize the standardized Channel First data (14, H, W)
        # Output will be (14, 42, 42) to match standard expectation, 
        # OR should we match the input format? 
        # User SR script handles both, so strict (14, 42, 42) is safer/cleaner.
        raw_img_lr = np.zeros((14, 42, 42), dtype=raw_img_c_first.dtype)
        for c in range(14):
            raw_img_lr[c] = cv2.resize(raw_img_c_first[c], (42, 42), interpolation=cv2.INTER_CUBIC)
        create_h5_from_array(os.path.join(DIRS['raw_h5_lr'], s, f"{save_id}.h5"), raw_img_lr)
        
        # --- 2. Process Image (Visual Parts) ---
        # rgb_raw is already (128, 128, 3) and normalized

        # [A4, B1] img_128_GT
        cv2.imwrite(os.path.join(DIRS['img_128_GT'], s, f"{save_id}.png"), to_vis_fair(rgb_raw))
        
        # [B2] img_384_Bic (128 -> 384)
        bic_384 = cv2.resize(rgb_raw, (384, 384), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(DIRS['img_384_Bic'], s, f"{save_id}.png"), to_vis_fair(bic_384))

        # [A1] img_42_LR (128 -> 42)
        lr_42 = cv2.resize(rgb_raw, (42, 42), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(DIRS['img_42_LR'], s, f"{save_id}.png"), to_vis_fair(lr_42))

        # [A2] img_128_Bic (128 -> 42 -> 128)
        # Simulate interpolation from LR back to original size
        bic_128 = cv2.resize(lr_42, (128, 128), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(DIRS['img_128_Bic'], s, f"{save_id}.png"), to_vis_fair(bic_128))
        
        # --- 3. Process Labels ---
        mask_path = os.path.join(TRAIN_MASK_DIR, basename.replace('image', 'mask'))
        if os.path.exists(mask_path):
            with h5py.File(mask_path, 'r') as f: mask_data = f['mask'][:]
            
            # Generate label content
            with h5py.File(mask_path, 'r') as f: mask_data = f['mask'][:]
            if mask_data.ndim == 3: mask_data = mask_data.squeeze()
            h_mask, w_mask = mask_data.shape
            
            contours, _ = cv2.findContours((mask_data>0).astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            txt_content = ""
            for cnt in contours:
                if cv2.contourArea(cnt) < 5: continue
                line = "0"
                for pt in cnt.reshape(-1, 2): line += f" {pt[0]/w_mask:.6f} {pt[1]/h_mask:.6f}"
                txt_content += line + "\n"
            
            # Save to Central Headers (Optional but good for backup)
            central_lbl_path = os.path.join(DIRS['labels'], s, f"{save_id}.txt")
            with open(central_lbl_path, 'w') as tf: tf.write(txt_content)
            
            # COPY Labels to EACH image folder's sibling 'labels' folder (YOLO Requirement)
            # Structure: 
            #   img_128_GT/train/xxx.png
            #   img_128_GT/train/xxx.txt  <-- YOLO looks here (or ../labels/xxx.txt)
            # Actually standard YOLO is:
            #   dataset/images/train/x.jpg
            #   dataset/labels/train/x.txt
            # But we have:
            #   yolo_data/img_128_GT/train/x.png
            # So we should put labels in:
            #   yolo_data/img_128_GT/labels/train/x.txt (if we treat img_128_GT as a dataset root)
            # OR just put .txt next to .png (simplest for custom config)
            
            # Let's put them in a 'labels' folder parallel to 'train/val/test' inside each dataset
            # Wait, DIRS keys are dataset roots? No, DIRS keys are specific folders.
            # DIRS['img_128_GT'] = .../img_128_GT
            # Inside is train/val/test.
            # So we need .../img_128_GT/labels/train/x.txt ? 
            # No, usually it's images/train and labels/train.
            # Our structure is: img_128_GT/train/x.png.
            # So if valid yaml points to img_128_GT/train, YOLO looks for img_128_GT/labels OR img_128_GT/../labels (which is yolo_data/labels).
            # The issue in previous run was "No labels found in .../img_128_GT/val.cache".
            # This implies it looked. 
            # If we simply put .txt files INSIDE `img_128_GT/train/`, YOLO will find them.
            # This is the most robust way for non-standard structures.
            
            # 確保如果 SR 資料夾已經存在（使用者先 finalize 過再跑 prepare），裡面的 Label 也會被強制更新
            label_targets = ['img_128_GT', 'img_384_Bic', 'img_42_LR', 'img_128_Bic']
            
            # 動態檢查 SR 資料夾是否存在，存在的話也加進去覆寫 Label
            dir_126_sr = os.path.join(OUTPUT_ROOT, 'yolo_data/img_126_SR')
            dir_384_sr = os.path.join(OUTPUT_ROOT, 'yolo_data/img_384_SR')
            
            if os.path.exists(dir_126_sr):
                DIRS['img_126_SR'] = dir_126_sr
                label_targets.append('img_126_SR')
            if os.path.exists(dir_384_sr):
                DIRS['img_384_SR'] = dir_384_sr
                label_targets.append('img_384_SR')

            for key in label_targets:
                # Save label right next to image
                # Construct path: DIRS[key] / s / filename.txt
                tgt_dir = os.path.join(DIRS[key], s)
                # 只有當該 split (train/val/test) 資料夾存在時才寫入（防呆）
                if os.path.exists(tgt_dir):
                    tgt_path = os.path.join(tgt_dir, f"{save_id}.txt")
                    with open(tgt_path, 'w') as tf: tf.write(txt_content)

if __name__ == "__main__": process()