import os
print("DEBUG: Script started", flush=True)
import argparse
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
import piq
from thop import profile
import cv2
from datetime import datetime

# 引用模組
from models.network_fusion import NGswin_MoSE
from models.network_swinir import SwinIR
from utils.losses import FusionLoss
from data.dataset import DIV2KDatasetRandomCrop, DIV2KDatasetVal

def set_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def prepare_image_for_wandb(tensor, clip_max=1.0, target_size=None, log_name="Image"):
    """
    WandB Image Preparation
    Since we are now using Pre-processed PNGs (0-255 -> 0-1 float),
    clip_max should be 1.0 to display the full range.
    """
    # 1. To Numpy (C, H, W)
    if tensor.ndim == 4:
        img = tensor[0].detach().cpu().numpy()
    else:
        img = tensor.detach().cpu().numpy()

    # 2. Handle Multispectral (Keep first 3)
    if img.shape[0] > 3:
        img = img[:3, :, :]

    # 3. Transpose (C, H, W) -> (H, W, C)
    img = np.transpose(img, (1, 2, 0))

    # 4. Handle Grayscale
    if img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=2)

    # Debug Print
    print(f"[{log_name}] Raw Shape: {img.shape}, Min: {img.min():.4f}, Max: {img.max():.4f}")

    # 5. Normalization
    if clip_max is not None:
        img = np.clip(img, 0, clip_max)
        img = img / clip_max 
        # Add Gamma Correction (2.2) for Visualization matching inspect_dataset.py
        img = np.power(img, 1.0/2.2)
        img = img * 255.0
    else:
        # Fallback
        img = (img - img.min()) / (img.max() - img.min()) * 255.0
        
    img = np.round(img).clip(0, 255).astype(np.uint8)
    
    # 6. Resize LR if needed (Nearest Neighbor)
    if target_size is not None:
        if img.shape[0] != target_size[0] or img.shape[1] != target_size[1]:
            img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)

    return wandb.Image(img, caption=f"{log_name} (Clip:{clip_max})")

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, clip_val=1.0):
    model.train()
    avg_loss = 0
    avg_psnr = 0
    avg_ssim = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    for i, (hr, lr) in enumerate(pbar):
        hr, lr = hr.to(device), lr.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        # Forward
        output = model(lr)
        if isinstance(output, tuple):
            sr, moe_loss = output
        else:
            sr = output
            moe_loss = torch.tensor(0.0, device=device)
        
        # Loss
        loss, loss_dict = criterion(sr, hr, moe_loss)
        
        loss.backward()
        optimizer.step()
        
        avg_loss += loss.item()
        
        # Metrics
        with torch.no_grad():
            sr_clamped = sr.detach()
            hr_clamped = hr.detach()
            
            if clip_val is not None:
                sr_clamped = sr_clamped / clip_val
                hr_clamped = hr_clamped / clip_val
                
            sr_clamped = sr_clamped.clamp(0, 1)
            hr_clamped = hr_clamped.clamp(0, 1)

            avg_psnr += piq.psnr(sr_clamped, hr_clamped, data_range=1.0).item()
            avg_ssim += piq.ssim(sr_clamped, hr_clamped, data_range=1.0).item()

        if (i + 1) % 100 == 0:
            wandb.log({
                "train/step_loss": loss.item(),
                **{f"train/loss_{k}": v for k, v in loss_dict.items()}
            })

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "psnr": f"{avg_psnr / (i + 1):.2f}",
            "ssim": f"{avg_ssim / (i + 1):.4f}"
        })
        
    return avg_loss / len(loader), avg_psnr / len(loader), avg_ssim / len(loader)

def validate(model, loader, device, clip_val=1.0):
    model.eval()
    avg_psnr = 0
    avg_ssim = 0
    
    with torch.no_grad():
        for i, (hr, lr, filename) in enumerate(loader):
            hr, lr = hr.to(device), lr.to(device)
            output = model(lr)
            if isinstance(output, tuple):
                sr, _ = output
            else:
                sr = output
            
            if i == 0:
                print(f"[Validate] Logging image: {filename[0]}")
                target_h, target_w = hr.shape[2], hr.shape[3]
                
                # Visualization Clip: 0.3 (to make images visible/brighter for human eye)
                # Training/Metrics Clip: 1.0 (Full Range)
                vis_clip = 0.3
                
                wandb.log({
                    "PNG_Check/HR": prepare_image_for_wandb(hr, clip_max=vis_clip, log_name="HR"),
                    "PNG_Check/LR": prepare_image_for_wandb(lr, clip_max=vis_clip, target_size=(target_h, target_w), log_name="LR"),
                    "PNG_Check/SR": prepare_image_for_wandb(sr, clip_max=vis_clip, log_name="SR")
                }, commit=True)

            if clip_val is not None:
                sr = sr / clip_val
                hr = hr / clip_val
            
            sr = sr.clamp(0, 1)
            hr = hr.clamp(0, 1)
            
            avg_psnr += piq.psnr(sr, hr, data_range=1.0).item()
            avg_ssim += piq.ssim(sr, hr, data_range=1.0).item()
            
    return avg_psnr / len(loader), avg_ssim / len(loader)

def get_args():
    parser = argparse.ArgumentParser()
    # Update defaults to Self-Supervised folders (dataset_self_supervised)
    # Note: Relative to where train.py is executed
    parser.add_argument('--data_hr', type=str, default='./dataset_self_supervised/train_hr')
    parser.add_argument('--data_lr', type=str, default='./dataset_self_supervised/train_lr')
    # Use Test set for Validation monitoring (or clean Val set)
    parser.add_argument('--val_hr', type=str, default='./dataset_self_supervised/val_hr')
    parser.add_argument('--val_lr', type=str, default='./dataset_self_supervised/val_lr')
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    # Default clip to 1.0 because data is already normalized to 0-1 (from 0-255 PNG)
    parser.add_argument('--clip', type=float, default=1.0, help='Clip value (e.g., 1.0 for PNG)')
    parser.add_argument('--patch_size', type=int, default=128, help='Training Patch Size (LR). HR size will be patch_size * scale')
    parser.add_argument('--use_moe', action='store_true')
    parser.add_argument('--experts', type=int, default=8)
    parser.add_argument('--wandb_project', type=str, default='NGram_MoSE_Fusion')
    parser.add_argument('--run_name', type=str, default='NGram_MoSE_Fusion_TIF', help='Name of the WandB run')
    parser.add_argument('--wandb_entity', type=str, default=None)
    
    # Loss Weights
    parser.add_argument('--w_pixel', type=float, default=1.0, help='Weight for L1 Pixel Loss')
    parser.add_argument('--w_ssim', type=float, default=1.0, help='Weight for SSIM Loss')
    parser.add_argument('--w_ncc', type=float, default=0.5, help='Weight for NCC Loss')
    parser.add_argument('--w_perceptual', type=float, default=0.1, help='Weight for VGG Perceptual Loss')
    parser.add_argument('--w_moe', type=float, default=0.2, help='Weight for MoE Load Balancing Loss')
    
    # Model Selection
    parser.add_argument('--model_type', type=str, default='ngram', choices=['ngram', 'swinir'], help='Model type: ngram or swinir')
    
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_deterministic(100)
    
    if args.wandb_project:
        # Auto-append timestamp if using default run name to prevent confusion
        if args.run_name == 'NGram_MoSE_Fusion_TIF':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.run_name = f"{args.run_name}_{timestamp}"
            
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args, name=args.run_name)
    
    print(f"Loading data... (Scale: x{args.scale}, Clip: {args.clip})")
    print("DEBUG: High Precision Training Mode Enabled. Using Float32 tensors (0-1 range) normalized from original bit depth.")
    print("DEBUG: PSNR will be calculated on full float precision.")
    print("DEBUG: WandB images are converted to 8-bit for display only.")
    
    train_set = DIV2KDatasetRandomCrop(
        root_hr=args.data_hr,
        root_lr=args.data_lr,
        crop_size=args.patch_size,
        scale=args.scale,
        clip_value=args.clip
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    val_set = DIV2KDatasetVal(
        root_hr=args.val_hr,
        root_lr=args.val_lr,
        clip_value=args.clip
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    if args.model_type == 'swinir':
        print(f"Creating Model: SwinIR (Classical SR)...")
        # Standard SwinIR parameters for Classical SR
        model = SwinIR(
            upscale=args.scale,
            in_chans=3,
            img_size=args.patch_size,
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='1conv'
        ).to(device)
    else:
        print(f"Creating Model: NGswin-MoSE (Experts={args.experts})...")
        model = NGswin_MoSE(
            upscale=args.scale, 
            use_moe=args.use_moe, 
            moe_experts=args.experts
        ).to(device)
    
    try:
        print("="*50)
        print(" [Model Metrics Calculation]")
        dummy = torch.randn(1, 3, 64, 64).to(device)
        flops, params = profile(model, inputs=(dummy,), verbose=False)
        print(f" Input Shape: (1, 3, 64, 64)")
        print(f" Params: {params/1e6:.2f}M")
        print(f" FLOPs : {flops/1e9:.2f}G")
        print("="*50)
        if wandb.run:
            wandb.log({"Params (M)": params/1e6, "FLOPs (G)": flops/1e9})
    except Exception as e:
        print(f"Metrics Error: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = FusionLoss(
        w_pixel=args.w_pixel, 
        w_ssim=args.w_ssim, 
        w_ncc=args.w_ncc, 
        w_moe=args.w_moe,
        w_perceptual=args.w_perceptual
    ).to(device)
    
    
    # Checkpoints: save in separate folders based on run_name to avoid overwrite
    save_dir = os.path.join("checkpoints", args.run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {save_dir}")
    
    print("Running initial validation...")
    validate(model, val_loader, device, args.clip)

    for epoch in range(args.epochs):
        loss, t_psnr, t_ssim = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, args.clip)
        v_psnr, v_ssim = validate(model, val_loader, device, args.clip)
        
        print(f"Epoch {epoch+1}: Loss={loss:.4f} | Train PSNR={t_psnr:.2f} | Val PSNR={v_psnr:.2f}")
        
        wandb.log({
            "train/loss": loss,
            "train/psnr": t_psnr,
            "val/psnr": v_psnr,
            "val/ssim": v_ssim,
            "epoch": epoch + 1
        })
        
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    main()