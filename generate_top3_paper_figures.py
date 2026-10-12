import os
import glob
import argparse
import cv2
import torch
import math
import numpy as np
import rasterio

from models.network_fusion import NGswin_MoSE
from models.network_swinir import SwinIR
from utils.checkpoint import load_checkpoint
import piq
import warnings
warnings.filterwarnings('ignore')

VIS_MAX = 0.3
GAMMA = 2.2

def get_args():
    parser = argparse.ArgumentParser(description="Paper Fig. 4: export the top-3 OOD scenes as separate images")
    parser.add_argument('--input_dir', type=str, default='dataset_40_OOD_testset')
    parser.add_argument('--output_dir', type=str, default='dataset_top3_OOD_paper_figures')
    parser.add_argument('--ckpt_ours', type=str, default='checkpoints/ngram_mose_x3_e500.pth')
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
    # 使用 piq 庫，傳入 CxHxW torch tensor，數值 0-1
    t1 = torch.from_numpy(img1.transpose(2, 0, 1)).unsqueeze(0).float()
    t2 = torch.from_numpy(img2.transpose(2, 0, 1)).unsqueeze(0).float()
    return piq.ssim(t1, t2, data_range=1.0).item()

def main():
    args = get_args()
    INPUT_DIR, OUTPUT_DIR = args.input_dir, args.output_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用硬體: {device}")

    # 載入模型
    print("\n[1/2] 載入 NGram-MoSE (Ours)...")
    model_ours = NGswin_MoSE(upscale=3, use_moe=True, moe_experts=8).to(device)
    print(f"-> 載入權重: {args.ckpt_ours}")
    model_ours = load_checkpoint(model_ours, args.ckpt_ours, device)

    print("\n[2/2] 載入 SwinIR (Baseline)...")
    model_swinir = SwinIR(upscale=3, in_chans=3, img_size=160, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv').to(device)
    print(f"-> 載入權重: {args.ckpt_swinir}")
    model_swinir = load_checkpoint(model_swinir, args.ckpt_swinir, device, is_swinir=True)

    BAD_REGIONS = ["Urban_Berlin", "Urban_RioDeJaneiro", "Urban_Athens", "Forest_CongoBasin", "Urban_SanFrancisco", "Urban_LosAngeles"]
    val_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.tif")))
    val_files = [f for f in val_files if not any(b in f for b in BAD_REGIONS)]
    
    if not val_files:
        print("找不到影像！")
        return

    print("\n開始掃描所有影像以找出表現最好的前3名...")
    results_list = []
    
    for tif_path in val_files:
        name = os.path.basename(tif_path).split('.')[0]
        with rasterio.open(tif_path) as src:
            h, w = src.height, src.width
            cx, cy = w // 2, h // 2
            size = 240
            data = src.read((1, 2, 3), window=rasterio.windows.Window(cx - size, cy - size, size * 2, size * 2))
            
        img_hr_raw = data.transpose(1, 2, 0).astype(np.float32)
        if img_hr_raw.max() > 255:
            if np.percentile(img_hr_raw, 1) > 800: img_hr_raw = np.clip(img_hr_raw - 1000.0, 0, None)
            img_hr_raw = img_hr_raw / 10000.0
        else:
            img_hr_raw = img_hr_raw / 255.0
            
        H, W, _ = img_hr_raw.shape
        new_H, new_W = H // 3, W // 3
        img_hr_float = img_hr_raw[:new_H * 3, :new_W * 3, :]
        img_lr_float = cv2.resize(img_hr_float, (new_W, new_H), interpolation=cv2.INTER_AREA)
        
        # Inference
        lr_tensor = torch.from_numpy(img_lr_float.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            img_bicubic_float = cv2.resize(img_lr_float, (new_W * 3, new_H * 3), interpolation=cv2.INTER_CUBIC).clip(0, 1)
            sr_tensor_ours, _ = model_ours(lr_tensor)
            img_ours_float = sr_tensor_ours.clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
            sr_tensor_swinir = model_swinir(lr_tensor)
            img_swinir_float = sr_tensor_swinir.clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
            
        # 因應 Sentinel-2 可能有極端反光像素超過 1.0 (導致 SSIM 函式庫報錯)
        # 強制裁切以符合 0-1 的評估標準
        img_hr_float = img_hr_float.clip(0, 1)
        img_bicubic_float = img_bicubic_float.clip(0, 1)
        
        # Metrics 算分
        p_b = calc_psnr(img_bicubic_float, img_hr_float)
        p_swin = calc_psnr(img_swinir_float, img_hr_float)
        p_our = calc_psnr(img_ours_float, img_hr_float)
        
        s_b = calc_ssim(img_bicubic_float, img_hr_float)
        s_swin = calc_ssim(img_swinir_float, img_hr_float)
        s_our = calc_ssim(img_ours_float, img_hr_float)
        
        # 紀錄結果與對應的視覺化陣列
        # 將視覺化的 BGR 存起來，避免等下再跑一次
        vis_lr = cv2.resize(to_vis(img_lr_float), (new_W * 3, new_H * 3), interpolation=cv2.INTER_NEAREST)
        results_list.append({
            'name': name,
            'psnr_ours': p_our, 'psnr_swinir': p_swin, 'psnr_bicubic': p_b,
            'ssim_ours': s_our, 'ssim_swinir': s_swin, 'ssim_bicubic': s_b,
            # 準備好 5 種類型的圖檔 (轉為 BGR 供 OpenCV 存檔)
            'img_dict': {
                '1_LR': cv2.cvtColor(vis_lr, cv2.COLOR_RGB2BGR),
                '2_Bicubic': cv2.cvtColor(to_vis(img_bicubic_float), cv2.COLOR_RGB2BGR),
                '3_SwinIR': cv2.cvtColor(to_vis(img_swinir_float), cv2.COLOR_RGB2BGR),
                '4_NGramMoSE_Ours': cv2.cvtColor(to_vis(img_ours_float), cv2.COLOR_RGB2BGR),
                '5_HR_GT': cv2.cvtColor(to_vis(img_hr_float), cv2.COLOR_RGB2BGR)
            }
        })
        print(f"[{name}] Ours PSNR: {p_our:.2f} (贏 Bicubic: {p_our - p_b:+.2f}dB)")

    # 依照 (Ours的分數 - Bicubic的分數) 的差距做降序排列，選出進步最大的前3名
    results_list.sort(key=lambda x: (x['psnr_ours'] - x['psnr_bicubic']), reverse=True)
    top_3 = results_list[:3]
    
    # 輸出資料夾準備
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n=========================================")
    print(f"🎉 成功挑選出最佳 3 張 OOD 展示圖，準備輸出...")
    
    for i, res in enumerate(top_3):
        rank = i + 1
        name = res['name']
        region_dir = os.path.join(OUTPUT_DIR, f"Top{rank}_{name}")
        os.makedirs(region_dir, exist_ok=True)
        
        # 1. 輸出 5 張獨立且無文字的圖片
        for tag, img_bgr in res['img_dict'].items():
            img_path = os.path.join(region_dir, f"{tag}.png")
            cv2.imwrite(img_path, img_bgr)
            
        # 2. 輸出 PSNR/SSIM 獨立成績單 txt
        txt_path = os.path.join(region_dir, "metrics.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Region: {name}\n")
            f.write("="*30 + "\n")
            f.write(f"【Bicubic】\nPSNR: {res['psnr_bicubic']:.4f} dB\nSSIM: {res['ssim_bicubic']:.4f}\n\n")
            f.write(f"【SwinIR (Baseline)】\nPSNR: {res['psnr_swinir']:.4f} dB\nSSIM: {res['ssim_swinir']:.4f}\n\n")
            f.write(f"【NGram-MoSE (Ours)】\nPSNR: {res['psnr_ours']:.4f} dB\nSSIM: {res['ssim_ours']:.4f}\n")
            
        print(f"-> 已輸出 Top {rank} ({name}) 到 {region_dir}")
        print(f"   (Bicubic: {res['psnr_bicubic']:.2f}, SwinIR: {res['psnr_swinir']:.2f}, Ours: {res['psnr_ours']:.2f})")
        
    print(f"=========================================")
    print(f"完成！所有圖片與成績單 txt 檔案已存放在 {OUTPUT_DIR} 資料夾。")

if __name__ == "__main__":
    main()
