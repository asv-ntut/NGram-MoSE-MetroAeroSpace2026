import os
import torch
import cv2
import numpy as np
import argparse
import glob
import rasterio
from models.network_fusion import NGswin_MoSE
from utils.checkpoint import load_checkpoint

# ================= Configuration =================
# Model Settings
EXPERTS = 8
SCALE = 3

# Visualization
VIS_MAX = 0.3
GAMMA = 2.2

def load_tif(path):
    with rasterio.open(path) as src:
        raw = src.read() # (C, H, W)
        
    # Handle Bands (Take first 3 if >3)
    if raw.shape[0] > 3:
        raw = raw[:3, :, :]
        
    img = raw.transpose(1, 2, 0) # (H, W, C)
    
    # Normalization (Raw 16-bit -> 0-1 Float)
    if img.dtype == np.uint16 or img.max() > 255:
        img = img.astype(np.float32) / 10000.0
    else:
        # Fallback for 8-bit
        img = img.astype(np.float32) / 255.0
        
    return img

def to_vis(img_float):
    """Convert 0-1 float to 8-bit for saving."""
    img = np.clip(img_float, 0, VIS_MAX)
    img = img / VIS_MAX
    img = np.power(img, 1.0/GAMMA)
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img[:, :, ::-1] # RGB to BGR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input image (LR) or Directory')
    parser.add_argument('--output', type=str, default=None, help='Output path or directory')
    parser.add_argument('--ckpt', type=str, default='checkpoints/ngram_mose_x3_e500.pth', help='NGram-MoSE checkpoint file')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Model
    print(f" -> Loading Model: NGram MoSE (Experts={EXPERTS})")
    print(f" -> Checkpoint: {args.ckpt}")

    model = NGswin_MoSE(upscale=SCALE, use_moe=True, moe_experts=EXPERTS).to(device)
    model = load_checkpoint(model, args.ckpt, device)
    
    # 2. Determine Files to Process
    files = []
    is_batch = False
    
    if os.path.isdir(args.input):
        is_batch = True
        print(f"Detected Directory Input: {args.input}")
        extensions = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']
        for ext in extensions:
            files.extend(glob.glob(os.path.join(args.input, ext)))
        
        if not files:
            print("No images found in directory!")
            return
            
        # Set Default Output Directory for Batch
        if args.output is None:
            args.output = "inference_results"
        os.makedirs(args.output, exist_ok=True)
        print(f" -> Output Directory: {args.output}")
        
    else:
        files = [args.input]
        # Set Default Output File for Single
        if args.output is None:
            args.output = "output_sr.png"

    # Metrics Accumulators
    total_psnr = 0.0
    total_ssim = 0.0
    count_metrics = 0
    
    print(f"Starting Inference on {len(files)} files...")
    
    # 3. Batch Loop
    for idx, file_path in enumerate(files):
        filename = os.path.basename(file_path)
        
        # Determine Output Path
        if is_batch:
            name_no_ext = os.path.splitext(filename)[0]
            save_path = os.path.join(args.output, f"{name_no_ext}_sr.png")
        else:
            save_path = args.output
            
        print(f"Processing ({idx+1}/{len(files)}): {filename}")

        # Load Data
        if file_path.lower().endswith(('.tif', '.tiff')):
            img_lr = load_tif(file_path)
        else:
            img_lr = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img_lr is None:
                print(f"  [Error] Could not read {file_path}")
                continue
            img_lr = img_lr[:, :, ::-1].astype(np.float32) / 255.0
            
        # To Tensor
        input_tensor = torch.from_numpy(img_lr.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        
        # Inference
        import piq
        with torch.no_grad():
            output = model(input_tensor)
            if isinstance(output, tuple): output = output[0]
            
        # Metrics Calculation
        hr_path = file_path.replace("test_lr", "test_hr").replace("train_lr", "train_hr").replace("val_lr", "val_hr")
        
        has_hr = False
        if os.path.exists(hr_path) and hr_path != file_path:
            has_hr = True
            if hr_path.lower().endswith(('.tif', '.tiff')):
                img_hr = load_tif(hr_path)
            else:
                 img_hr = cv2.imread(hr_path, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0
                 
            hr_tensor = torch.from_numpy(img_hr.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
            
            # Dimensions Check
            if hr_tensor.shape != output.shape:
                hr_tensor = torch.nn.functional.interpolate(hr_tensor, size=output.shape[2:], mode='bicubic')
                
            psnr = piq.psnr(output.clamp(0, 1), hr_tensor.clamp(0, 1), data_range=1.0).item()
            ssim = piq.ssim(output.clamp(0, 1), hr_tensor.clamp(0, 1), data_range=1.0).item()
            
            total_psnr += psnr
            total_ssim += ssim
            count_metrics += 1
            
            print(f"  -> PSNR: {psnr:.2f} | SSIM: {ssim:.4f}")
        
        # Save Result
        sr_img = output.squeeze(0).cpu().numpy().transpose(1, 2, 0) # (H, W, C)
        out_bgr = to_vis(sr_img)
        cv2.imwrite(save_path, out_bgr)
        
        # [NEW] Save Input (LR) and Ground Truth (HR) for Comparison
        lr_save_path = save_path.replace("_sr.png", "_lr.png")
        lr_bgr = to_vis(img_lr)
        cv2.imwrite(lr_save_path, lr_bgr)
        
        if has_hr:
            hr_save_path = save_path.replace("_sr.png", "_hr.png")
            hr_bgr = to_vis(img_hr)
            cv2.imwrite(hr_save_path, hr_bgr)
        
    # 4. Final Summary
    print("\n" + "="*30)
    print(" Inference Completed!")
    if count_metrics > 0:
        avg_psnr = total_psnr / count_metrics
        avg_ssim = total_ssim / count_metrics
        print(f" Average PSNR: {avg_psnr:.4f} dB")
        print(f" Average SSIM: {avg_ssim:.4f}")
    else:
        print(" No Ground Truth found for metrics.")
    print(f" Results (LR/SR/HR) saved to: {args.output}")
    print("="*30)

if __name__ == "__main__":
    main()
