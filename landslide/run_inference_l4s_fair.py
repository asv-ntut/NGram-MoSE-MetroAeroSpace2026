import os
import argparse
import glob
import torch
import cv2
import numpy as np
import h5py
import sys
from tqdm import tqdm

# Ensure models can be imported
sys.path.append(os.getcwd())
try:
    from models.network_fusion import NGswin_MoSE
    from models.network_swinir import SwinIR  # Import SwinIR
    from utils.checkpoint import load_checkpoint
except ImportError:
    print("Error: Could not import models. Ensure you are in the correct directory.")
    sys.exit(1)

# ================= Configuration =================
# We no longer hardcode CHECKPOINT_DIR here so you can pass different models easily
EXPERTS = 8
SCALE = 3

# Visualization Parameters (FAIR comparison - applied equally to all models)
VIS_MAX = 0.15   # Decreased from 0.3 to make images brighter globally.
GAMMA = 1.2      # Adjusted gamma to boost dark details fairly across all models.

def get_latest_checkpoint(folder):
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Checkpoint folder not found: {folder}")
    files = glob.glob(os.path.join(folder, "*.pth"))
    if not files:
        raise FileNotFoundError(f"No .pth files found in {folder}")
    def extract_epoch(f):
        try: return int(f.split('_epoch_')[-1].split('.')[0])
        except: return 0
    latest = max(files, key=extract_epoch)
    return latest, extract_epoch(latest)

def to_vis_fair(img_float):
    """
    Fixed/Fair Visualization function.
    No adaptive stretching. Every model gets judged by the exact same ruler.
    """
    img = np.clip(img_float, 0, VIS_MAX)
    img = img / VIS_MAX
    img = np.power(img, 1.0/GAMMA)
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img[:, :, ::-1] # BGR for cv2 save

def load_data_and_norm(path):
    if path.lower().endswith('.h5'):
        with h5py.File(path, 'r') as f:
            if 'img' in f: data = f['img'][:]
            elif 'data' in f: data = f['data'][:]
            else: data = f['image'][:]
        if data.ndim == 3 and data.shape[0] == 14: img = np.stack([data[3], data[2], data[1]], axis=2)
        elif data.ndim == 3 and data.shape[2] == 14: img = data[:, :, [3, 2, 1]]
    else:
        ret, imgs = cv2.imreadmulti(path, flags=cv2.IMREAD_UNCHANGED)
        if len(imgs) >= 4:
             b4, b3, b2 = imgs[3], imgs[2], imgs[1]
             img = np.stack([b4, b3, b2], axis=2)
        elif len(imgs) == 3 or path.lower().endswith(('.png', '.jpg')):
             img = cv2.imread(path, cv2.IMREAD_COLOR)
             if img is None: return None
             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
             print(f"Unsupported image format or bands in {path}")
             return None

    # Normalization (Matches L4S Official Training Standard)
    if img.dtype == np.uint16 or img.max() > 255: 
        img = img.astype(np.float32) / 10000.0
    elif img.max() > 1.0: 
        # L4S OFFICIAL FACTORS!
        factors = np.array([0.04452, 0.06634, 0.08246], dtype=np.float32)
        img = img.astype(np.float32) * factors
    else: 
        img = img.astype(np.float32)
    
    img = np.clip(img, 0, 1)
    return img

def fit_to_patch_size(tensor, patch_size=8):
    # Padding to be multiple of patch_size (SwinIR needs 8x8 usually)
    _, _, h, w = tensor.shape
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    tensor = torch.nn.functional.pad(tensor, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return tensor, mod_pad_h, mod_pad_w

def main():
    parser = argparse.ArgumentParser(description="FAIR Inference for L4S YOLO Segmentation")
    parser.add_argument('--input', type=str, required=True, help="Path to input file or directory")
    parser.add_argument('--output', type=str, default='SR_temp', help="Root output folder")
    parser.add_argument('--model_type', type=str, choices=['ngram', 'swinir'], default='ngram', help="Which model architecture to load")
    parser.add_argument('--ckpt_dir', type=str, required=True, help="Path to the folder containing the .pth checkpoints")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Checkpoint
    try:
        if args.ckpt_dir.endswith('.pth'):
            ckpt_path = args.ckpt_dir
            print(f"Using defined Checkpoint file: {ckpt_path}")
        else:
            ckpt_path, epoch = get_latest_checkpoint(args.ckpt_dir)
            print(f"Loading Model: {os.path.basename(ckpt_path)} (Epoch {epoch})")
    except FileNotFoundError as e:
        print(e)
        return

    # 2. Initialize the correct Model Architecture
    if args.model_type == 'ngram':
        model = NGswin_MoSE(upscale=SCALE, use_moe=True, moe_experts=EXPERTS).to(device)
    elif args.model_type == 'swinir':
        # FIX: Your training command used --patch_size 64, so img_size must be 64!
        # FIX: The positional indices crashing (225x6 vs 961x6) proves window_size=8
        model = SwinIR(
            upscale=3, in_chans=3, img_size=64, window_size=8, img_range=1.,
            depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv'
        ).to(device)

    model = load_checkpoint(model, ckpt_path, device, is_swinir=(args.model_type == 'swinir'))

    # 3. Gather files
    files = []
    if not os.path.exists(args.input):
        print(f"Error: The input path '{args.input}' does not exist.")
        return

    if os.path.isdir(args.input):
        for root, _, fnames in os.walk(args.input):
            for fname in fnames:
                if fname.lower().endswith(('.tif', '.tiff', '.h5', '.png', '.jpg')): 
                    files.append(os.path.join(root, fname))
    else: 
        files = [args.input]

    if not files:
        print(f"No valid image files found in {args.input}")
        return

    print(f"Found {len(files)} files. Starting FAIR inference with [{args.model_type.upper()}] to {args.output}...")

    # 4. Inference loop mimicking the Original Dynamic Output Logic
    for i, path in enumerate(tqdm(files)):
        try:
            filename = os.path.basename(path)
            
            # --- Dynamic Output Path Logic (Same as original run_inference_l4s.py) ---
            out_res_folder = "384" 
            if "raw_h5_lr" in path.replace("\\", "/"):
                out_res_folder = "126" # Low Res Input -> 42*3 = 126
            elif "raw_h5" in path.replace("\\", "/"):
                out_res_folder = "384" # High Res Input -> 128*3 = 384
            
            subset = "test" 
            if "train" in path.replace("\\", "/"): subset = "train"
            elif "val" in path.replace("\\", "/"): subset = "val"
            elif "test" in path.replace("\\", "/"): subset = "test"
            
            # Construct final directory: e.g., SR_temp/126/test
            save_dir = os.path.join(args.output, out_res_folder, subset)
            os.makedirs(save_dir, exist_ok=True)
            
            # Filename
            save_name = f"{os.path.splitext(filename)[0]}_SR.png"
            sr_path = os.path.join(save_dir, save_name)
            
            # --- 1. Load Data (NO GAUSSIAN BLUR!) ---
            img_in = load_data_and_norm(path)
            if img_in is None: continue

            # To Tensor 
            img_in_gpu = torch.from_numpy(img_in.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            
            # Important for SWINIR specifically: padding to fit window size (Now 8x8 as per your checkpoint)
            img_in_gpu, pad_h, pad_w = fit_to_patch_size(img_in_gpu, patch_size=8)

            # --- 2. Inference (NO TEST-TIME AUGMENTATION!) ---
            with torch.no_grad():
                out = model(img_in_gpu)
                if isinstance(out, tuple): 
                    out = out[0]
            
            # Remove padding if it was added
            if pad_h > 0 or pad_w > 0:
                # the scale is 3, so padding on HR is pad_h*3, pad_w*3
                out = out[:, :, :-(pad_h*3) if pad_h>0 else None, :-(pad_w*3) if pad_w>0 else None]

            out_np = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            
            # --- 3. Save Fair Visualization ---
            vis_sr = to_vis_fair(out_np)
            cv2.imwrite(sr_path, vis_sr)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("\n" + "="*40)
    print(f" [{args.model_type.upper()}] FAIR INFERENCE COMPLETED!")
    print(f" Directory Structure: {args.output}/[Resolution]/[Subset]/")
    print("="*40)

if __name__ == "__main__":
    main()
