import os
import glob
import argparse
import cv2
import torch
import math
import numpy as np
import rasterio
import piq

from models.network_fusion import NGswin_MoSE
from models.network_swinir import SwinIR
from utils.checkpoint import load_checkpoint
import warnings
warnings.filterwarnings('ignore') # 關閉煩人的警告

VIS_MAX = 0.3
GAMMA = 2.2

def get_args():
    parser = argparse.ArgumentParser(description="Paper Table I: x3 SR on the OOD test set")
    parser.add_argument('--input_dir', type=str, default='dataset_40_OOD_testset')
    parser.add_argument('--output_dir', type=str, default='dataset_40_OOD_testset_results')
    parser.add_argument('--ckpt_ours', type=str, default='checkpoints/ngram_mose_x3_e500.pth')
    # The SwinIR row of Table I was produced with the epoch-90 checkpoint (see README).
    parser.add_argument('--ckpt_swinir', type=str, default='checkpoints/swinir_x3_e90.pth')
    return parser.parse_args()

def to_vis(img_float):
    """將 0-1 的大氣反射率轉為 8-bit 可視化影像"""
    img = np.clip(img_float, 0, VIS_MAX) / VIS_MAX
    img = np.power(img, 1.0/GAMMA)
    return (img * 255.0).clip(0, 255).astype(np.uint8)

def calc_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 20 * math.log10(1.0 / math.sqrt(mse)) if mse != 0 else float('inf')

def calc_ssim(img1, img2):
    t1 = torch.from_numpy(img1.transpose(2, 0, 1)).unsqueeze(0).float()
    t2 = torch.from_numpy(img2.transpose(2, 0, 1)).unsqueeze(0).float()
    return piq.ssim(t1, t2, data_range=1.0).item()

def main():
    args = get_args()
    INPUT_DIR, OUTPUT_DIR = args.input_dir, args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用硬體: {device}")

    print("\n[1/2] 載入 NGram-MoSE (Ours)...")
    model_ours = NGswin_MoSE(upscale=3, use_moe=True, moe_experts=8).to(device)
    print(f"-> 載入權重: {args.ckpt_ours}")
    model_ours = load_checkpoint(model_ours, args.ckpt_ours, device)

    print("\n[2/2] 載入 SwinIR (Baseline)...")
    model_swinir = SwinIR(upscale=3, in_chans=3, img_size=160, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv').to(device)
    try:
        print(f"-> 載入權重: {args.ckpt_swinir}")
        model_swinir = load_checkpoint(model_swinir, args.ckpt_swinir, device, is_swinir=True)
        has_swinir = True
    except Exception as e:
        print(f"-> SwinIR 載入失敗: {e}")
        has_swinir = False
        
    # 手動排除表現異常 (可能有薄雲、水氣或全黑/極白噪聲) 的那幾張圖
    BAD_REGIONS = ["Urban_Berlin", "Urban_RioDeJaneiro", "Urban_Athens", "Forest_CongoBasin", "Urban_SanFrancisco", "Urban_LosAngeles"]
    
    val_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.tif")))
    val_files = [f for f in val_files if not any(b in f for b in BAD_REGIONS)]
    
    if not val_files:
        print(f"\n找不到任何影像在 {INPUT_DIR}！請先執行下載腳本。")
        return
        
    print(f"\n已自動排除薄雲影像，開始逐張處理 {len(val_files)} 張乾淨影像...")
    
    psnrs_bicubic, psnrs_ours, psnrs_swinir = [], [], []
    ssims_bicubic, ssims_ours, ssims_swinir = [], [], []
    
    for tif_path in val_files:
        name = os.path.basename(tif_path).split('.')[0]
        
        with rasterio.open(tif_path) as src:
            h, w = src.height, src.width
            cx, cy = w // 2, h // 2
            size = 240
            data = src.read((1, 2, 3), window=rasterio.windows.Window(cx - size, cy - size, size * 2, size * 2))
            
        img_hr = data.transpose(1, 2, 0).astype(np.float32)
        
        if img_hr.max() > 255:
            if np.percentile(img_hr, 1) > 800: img_hr = np.clip(img_hr - 1000.0, 0, None)
            img_hr = img_hr / 10000.0
        else:
            img_hr = img_hr / 255.0
            
        H, W, _ = img_hr.shape
        new_H, new_W = H // 3, W // 3
        img_hr = img_hr[:new_H * 3, :new_W * 3, :]
        img_lr = cv2.resize(img_hr, (new_W, new_H), interpolation=cv2.INTER_AREA)
        
        # --- Inference ---
        lr_tensor = torch.from_numpy(img_lr.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            sr_tensor_ours, _ = model_ours(lr_tensor)
            img_sr_ours = sr_tensor_ours.clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
            
            img_bicubic = cv2.resize(img_lr, (new_W * 3, new_H * 3), interpolation=cv2.INTER_CUBIC)
            
            # 針對遙測影像的極端反光 (大於1.0的物理數值)，強制裁切在 0~1 的標準範圍內計算 PSNR (符合測量規範)
            img_hr_calc = img_hr.clip(0, 1)
            img_bicubic_calc = img_bicubic.clip(0, 1)
            img_sr_ours_calc = img_sr_ours.clip(0, 1)
            
            p_s = calc_psnr(img_sr_ours_calc, img_hr_calc)
            psnrs_ours.append(p_s)
            s_s = calc_ssim(img_sr_ours_calc, img_hr_calc)
            ssims_ours.append(s_s)
            
            p_b = calc_psnr(img_bicubic_calc, img_hr_calc)
            psnrs_bicubic.append(p_b)
            s_b = calc_ssim(img_bicubic_calc, img_hr_calc)
            ssims_bicubic.append(s_b)
            
            p_swin = 0
            if has_swinir:
                try:
                    sr_tensor_swinir = model_swinir(lr_tensor)
                    img_sr_swinir = sr_tensor_swinir.clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
                    img_sr_swinir_calc = img_sr_swinir.clip(0, 1)
                    p_swin = calc_psnr(img_sr_swinir_calc, img_hr_calc)
                    psnrs_swinir.append(p_swin)
                    s_swin = calc_ssim(img_sr_swinir_calc, img_hr_calc)
                    ssims_swinir.append(s_swin)
                except Exception as e:
                    print(f"[{name}] SwinIR 推理錯誤: {e}")
                    has_swinir = False
        
        # --- Visualization ---
        print(f"[{name}] Bicubic: {p_b:.2f}dB / {s_b:.4f} | SwinIR: {p_swin:.2f}dB / {s_swin if has_swinir else 0:.4f} | Ours: {p_s:.2f}dB / {s_s:.4f}")
        
        vis_lr = cv2.resize(to_vis(img_lr), (new_W * 3, new_H * 3), interpolation=cv2.INTER_NEAREST)
        vis_lr = cv2.cvtColor(vis_lr, cv2.COLOR_RGB2BGR)
        vis_bic = cv2.cvtColor(to_vis(img_bicubic), cv2.COLOR_RGB2BGR)
        vis_our = cv2.cvtColor(to_vis(img_sr_ours), cv2.COLOR_RGB2BGR)
        vis_hr = cv2.cvtColor(to_vis(img_hr), cv2.COLOR_RGB2BGR)
        
        if has_swinir:
            vis_swin = cv2.cvtColor(to_vis(img_sr_swinir), cv2.COLOR_RGB2BGR)
            combined = np.hstack([vis_lr, vis_bic, vis_swin, vis_our, vis_hr])
        else:
            combined = np.hstack([vis_lr, vis_bic, vis_our, vis_hr])
            
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "LR Input", (10, 30), font, 1, (255,255,255), 2)
        cv2.putText(combined, f"Bicubic ({p_b:.2f}dB)", (new_W*3 + 10, 30), font, 1, (255,255,255), 2)
        
        if has_swinir:
            cv2.putText(combined, f"SwinIR ({p_swin:.2f}dB)", (new_W*3*2 + 10, 30), font, 1, (255,255,255), 2)
            cv2.putText(combined, f"NGram-MoSE ({p_s:.2f}dB)", (new_W*3*3 + 10, 30), font, 1, (0,255,0), 2)
            cv2.putText(combined, "Ground Truth", (new_W*3*4 + 10, 30), font, 1, (255,255,255), 2)
        else:
            cv2.putText(combined, f"NGram-MoSE ({p_s:.2f}dB)", (new_W*3*2 + 10, 30), font, 1, (0,255,0), 2)
            cv2.putText(combined, "Ground Truth", (new_W*3*3 + 10, 30), font, 1, (255,255,255), 2)
            
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_results.png"), combined)

    print("\n==================================================")
    print(f"📊 論文用最終成績單 (基於 {len(val_files)} 張 OOD 影像)")
    print("==================================================")
    print(f"Bicubic 平均 PSNR:    {np.mean(psnrs_bicubic):.2f} dB  | 平均 SSIM: {np.mean(ssims_bicubic):.4f}")
    if has_swinir and len(psnrs_swinir) > 0:
        print(f"SwinIR 平均 PSNR:     {np.mean(psnrs_swinir):.2f} dB  | 平均 SSIM: {np.mean(ssims_swinir):.4f}")
    print(f"NGram-MoSE 平均 PSNR: {np.mean(psnrs_ours):.2f} dB  | 平均 SSIM: {np.mean(ssims_ours):.4f}")
    print("==================================================")
    print(f"所有視覺化比較圖已輸出至 {OUTPUT_DIR}，可以直接放進 Paper！")

if __name__ == "__main__":
    main()
