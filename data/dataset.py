import os
import glob
import random
import numpy as np
import cv2
import rasterio
import torch
from torch.utils.data import Dataset

class DIV2KDatasetRandomCrop(Dataset):
    def __init__(self, root_hr, root_lr, crop_size=64, scale=3, clip_value=1.0):
        self.crop_size = crop_size
        self.scale = scale
        self.clip_value = clip_value
        
        self.hr_files = sorted(glob.glob(os.path.join(root_hr, "*.tif"))) + sorted(glob.glob(os.path.join(root_hr, "*.TIF")))
        self.lr_files = sorted(glob.glob(os.path.join(root_lr, "*.tif"))) + sorted(glob.glob(os.path.join(root_lr, "*.TIF")))
        
        if len(self.hr_files) == 0:
            # Fallback to PNG or NPY
            self.hr_files = sorted(glob.glob(os.path.join(root_hr, "*.png"))) + sorted(glob.glob(os.path.join(root_hr, "*.npy")))
            self.lr_files = sorted(glob.glob(os.path.join(root_lr, "*.png"))) + sorted(glob.glob(os.path.join(root_lr, "*.npy")))

        if len(self.hr_files) == 0:
            print(f"Warning: No files (TIF/PNG/NPY) found in {root_hr}")
        if len(self.lr_files) == 0:
            print(f"Warning: No files (TIF/PNG/NPY) found in {root_lr}")
            
        # assert len(self.hr_files) == len(self.lr_files), f"HR ({len(self.hr_files)}) and LR ({len(self.lr_files)}) file counts do not match"

    def __len__(self):
        return min(len(self.hr_files), len(self.lr_files))

    def __getitem__(self, idx):
        # Load images
        hr_path = self.hr_files[idx]
        lr_path = self.lr_files[idx]
        
        def read_image(path):
            if path.lower().endswith(('.tif', '.tiff')):
                with rasterio.open(path) as src:
                    # rasterio reads as (C, H, W)
                    img = src.read()
                    # Convert to (H, W, C) for consistency with cv2/numpy pipeline
                    img = img.transpose(1, 2, 0)
                    
                    # Handle Data Type & Range
                    # Satellite data often needs specific normalization
                    # Here we mimic preprocess.py logic: if > 1.0, divide by 10000.0
                    # But wait, we usually want to return raw data and normalize later?
                    # The existing code expects standard image formats (H,W,C).
                    
                    # Let's keep it as float32 for high precision
                    img = img.astype(np.float32)
                    
                    return img
            elif path.endswith('.npy'):
                img = np.load(path)
                if len(img.shape) == 3:
                     # Check if it looks like BGR (from cv2 usually) -> Swap to RGB? 
                     # NPY from preprocess.py (old) was (C, H, W) or (H, W, C)? 
                     # preprocess.py saved PNG, so NPY loading was optional/legacy?
                     # Let's assume NPY is RGB if recently saved, or BGR if saved via cv2?
                     # Safest to assume it might be BGR if source was cv2. 
                     # But let's check simple fix first.
                     # The original code swapped BGR->RGB for NPY. We keep that.
                     img = img[:, :, ::-1].copy()
                return img
            else:
                # PNG / JPG
                # Use IMREAD_UNCHANGED to keep 16-bit
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise FileNotFoundError(f"Failed to load image: {path}")
                
                # BGR -> RGB
                # Note: valid for 8-bit and 16-bit 3-channel
                if len(img.shape) == 3:
                     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img

        hr_img = read_image(hr_path)
        lr_img = read_image(lr_path)

        # Random Crop
        # LR crop size is crop_size, HR crop size is crop_size * scale
        h_lr, w_lr, _ = lr_img.shape
        
        lr_crop_size = self.crop_size
        hr_crop_size = self.crop_size * self.scale
        
        # Ensure image is large enough
        if h_lr <= lr_crop_size or w_lr <= lr_crop_size:
             # If image is smaller than crop size, just resize it to crop size (not ideal but prevents crash)
             # Or just take the whole image if it matches?
             # For now, let's just center crop or resize if too small.
             # Simplest fallback: Resize to crop size + 1
             pass

        try:
            x = random.randint(0, w_lr - lr_crop_size)
            y = random.randint(0, h_lr - lr_crop_size)
        except ValueError:
            # Handle case where image is smaller than crop size
            x = 0
            y = 0
            # Resize to ensure it fits?
            lr_img = cv2.resize(lr_img, (max(w_lr, lr_crop_size), max(h_lr, lr_crop_size)))
            hr_img = cv2.resize(hr_img, (max(w_lr, lr_crop_size)*self.scale, max(h_lr, lr_crop_size)*self.scale))
        
        lr_crop = lr_img[y:y+lr_crop_size, x:x+lr_crop_size]
        hr_crop = hr_img[y*self.scale:(y+lr_crop_size)*self.scale, x*self.scale:(x+lr_crop_size)*self.scale]
        
        # Normalize to 0-1 and convert to Tensor (C, H, W)
        # Note: If NPY is already normalized, check this. Assuming NPY stores raw pixel values (e.g. 0-255 or 0-65535)
        
        # Helper to convert to tensor
        def to_tensor(img):
            # img: H, W, C
            # Handle grayscale
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
            
            if img.shape[2] > 3:
                img = img[:, :, :3]
            elif img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)

            return torch.from_numpy(img.transpose(2, 0, 1)).float()

        lr_tensor = to_tensor(lr_crop)
        hr_tensor = to_tensor(hr_crop)
        
        # Normalization
        # TIF (float32, potentially > 1.0)
        # 16-bit PNG (uint16, 0-65535)
        # 8-bit PNG (uint8, 0-255)
        
        def normalize_tensor(tensor, source_dtype=None):
            # If already small float (0-1), don't touch unless max > 1
             if tensor.max() > 1.0:
                 if source_dtype == np.uint16 or tensor.max() > 255.0:
                     # 16-bit or TIF (typically 0-10000 scale for Sentinel/Landsat reflectance)
                     # We use 10000.0 as divisor to keep values in ~0-1 range (can be >1)
                     return tensor / 10000.0
                 else:
                     # 8-bit
                     return tensor / 255.0
             return tensor

        lr_tensor = normalize_tensor(lr_tensor, lr_img.dtype)
        hr_tensor = normalize_tensor(hr_tensor, hr_img.dtype)
        
        return hr_tensor, lr_tensor

class DIV2KDatasetVal(Dataset):
    def __init__(self, root_hr, root_lr, clip_value=1.0):
        self.hr_files = sorted(glob.glob(os.path.join(root_hr, "*.tif"))) + sorted(glob.glob(os.path.join(root_hr, "*.TIF")))
        self.lr_files = sorted(glob.glob(os.path.join(root_lr, "*.tif"))) + sorted(glob.glob(os.path.join(root_lr, "*.TIF")))
        
        if len(self.hr_files) == 0:
            self.hr_files = sorted(glob.glob(os.path.join(root_hr, "*.png"))) + sorted(glob.glob(os.path.join(root_hr, "*.npy")))
            self.lr_files = sorted(glob.glob(os.path.join(root_lr, "*.png"))) + sorted(glob.glob(os.path.join(root_lr, "*.npy")))
        
        if len(self.hr_files) == 0:
             print(f"Warning: No files found in {root_hr}")
        self.clip_value = clip_value

    def __len__(self):
        return min(len(self.hr_files), len(self.lr_files))

    def __getitem__(self, idx):
        hr_path = self.hr_files[idx]
        lr_path = self.lr_files[idx]
        filename = os.path.basename(hr_path)
        
        def read_image(path):
            if path.lower().endswith(('.tif', '.tiff')):
                with rasterio.open(path) as src:
                    img = src.read()
                    img = img.transpose(1, 2, 0)
                    img = img.astype(np.float32)
                    return img
            elif path.endswith('.npy'):
                img = np.load(path)
                if len(img.shape) == 3:
                     img = img[:, :, ::-1].copy()
                return img
            else:
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise FileNotFoundError(f"Failed to load image: {path}")
                if len(img.shape) == 3:
                     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img

        hr_img = read_image(hr_path)
        lr_img = read_image(lr_path)
        
        def to_tensor(img):
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
            if img.shape[2] > 3:
                img = img[:, :, :3]
            elif img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            return torch.from_numpy(img.transpose(2, 0, 1)).float()

        lr_tensor = to_tensor(lr_img)
        hr_tensor = to_tensor(hr_img)
        
        # Reuse normalization logic from RandomCrop
        # Duplicated mainly for standalone use
        if lr_tensor.max() > 1.0:
             if lr_img.dtype == np.uint16 or lr_tensor.max() > 255.0:
                 lr_tensor /= 10000.0
             else:
                 lr_tensor /= 255.0

        if hr_tensor.max() > 1.0:
             if hr_img.dtype == np.uint16 or hr_tensor.max() > 255.0:
                 hr_tensor /= 10000.0
             else:
                 hr_tensor /= 255.0
        
        return hr_tensor, lr_tensor, filename
