# NGram-MoSE

Code for **NGram-MoSE: Efficient Remote Sensing Super-Resolution via N-Gram Context and Mixture-of-Experts**, 2026 IEEE 13th International Workshop on Metrology for AeroSpace (MetroAeroSpace), pp. 397–402.

**NGram-MoSE: Efficient Remote Sensing Super-Resolution via N-Gram Context and Mixture-of-Experts** 的官方程式碼，發表於 2026 IEEE MetroAeroSpace，pp. 397–402。

Yun-Hsuan Huang, Trong-An Bui, Chih-Hung Chuang<br>
Institute of Aerospace and Systems Engineering, National Taipei University of Technology（國立臺北科技大學 航空與系統工程研究所）

[[Paper (IEEE Xplore)]](https://ieeexplore.ieee.org/document/11646695) · DOI [10.1109/METROAEROSPACE69299.2026.11646695](https://doi.org/10.1109/METROAEROSPACE69299.2026.11646695)

This repository is the code release for the workshop paper (git tag `v1.0-metroaerospace2026`).

這個 repo 是 workshop 論文的程式碼，對應 git tag `v1.0-metroaerospace2026`。

NGram-MoSE is a lightweight Transformer for ×3 super-resolution of Sentinel-2 RGB imagery. Each N-Gram Swin Transformer Block (NSTB) combines:

- **N-Gram context injection** (n = 2): forward and backward sliding-window attention over neighbouring windows, added to the window tokens before window self-attention. This improves continuity across window boundaries.
- **A sparse Mixture-of-Experts FFN**: 8 experts with top-2 gating and a load-balancing loss, followed by a **Smart Merger** (3×3 depth-wise convolution + residual) that restores local spatial coherence.

NGram-MoSE 是用來做 Sentinel-2 RGB 影像 ×3 super-resolution 的輕量 Transformer。每個 N-Gram Swin Transformer Block（NSTB）包含：

- **N-Gram context injection**（n = 2）：對相鄰 window 做 forward 和 backward sliding-window attention，在 window self-attention 之前加到 window tokens 上，讓 window 邊界更連續。
- **Sparse Mixture-of-Experts FFN**：8 個 experts、top-2 gating 加上 load-balancing loss，最後接 **Smart Merger**（3×3 depth-wise convolution + residual），恢復局部的空間連續性。

The network has 14 NSTB blocks (stages of 6/4/4), embedding dimension 64, window size 8, and a pixel-shuffle reconstruction head.

網路共有 14 個 NSTB（三個 stage 分別是 6/4/4 個），embedding dimension 64，window size 8，最後用 pixel-shuffle 重建。

> **TODO before making this repository public:** choose a license (see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)), then delete this note.
>
> **TODO（公開前）：** 選定授權（見 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)），然後刪掉這段提醒。

## Results 實驗結果

**Table I: ×3 restoration on the out-of-distribution (OOD) test set** (29 scenes, 480×480 HR)

**Table I：OOD 測試集上的 ×3 重建結果**（29 個場景，HR 480×480）

| Method | PSNR (dB) | SSIM |
|---|---:|---:|
| Bicubic | 31.43 | 0.8873 |
| SwinIR | 31.31 | 0.8946 |
| **NGram-MoSE** | **31.68** | **0.9089** |

**Table II: complexity for a 128×128 LR input**

**Table II：模型複雜度**（LR 輸入 128×128）

| Method | Params (M) | FLOPs (G) |
|---|---:|---:|
| SwinIR | 11.89 | 195.34 |
| **NGram-MoSE** | **1.70** | **13.94** |

**Table III: Landslide4Sense, detector trained on native-resolution imagery** (YOLOv8s-seg, box mAP)

**Table III：Landslide4Sense，用原生解析度影像訓練的偵測器**（YOLOv8s-seg，box mAP）

| Input | mAP@50 | mAP@50-95 |
|---|---:|---:|
| Low-resolution input | 23.03% | 8.01%* |
| Bicubic | 25.64% | 9.49% |
| SwinIR | 25.61% | 8.47% |
| **NGram-MoSE** | **30.11%** | **10.49%** |
| Native-resolution reference | 43.88% | 19.26% |

\* The original evaluation log and a re-run both give **8.28%**. The 8.01% printed in the paper is a transcription error.

\* 原始的 evaluation log 和重跑結果都是 **8.28%**，論文上的 8.01% 是筆誤。

**Table IV: cross-scale robustness** (detector M2 trained on SR-upsampled 384×384 images; B1 native input, B2 bicubic-upsampled input, B3 SR-upsampled input)

**Table IV：cross-scale robustness**（偵測器 M2 用 SR 放大到 384×384 的影像訓練；B1 輸入原生解析度影像，B2 輸入 bicubic 放大影像，B3 輸入 SR 放大影像）

| Detector trained on 偵測器的訓練影像 | B1 mAP@50 / @50-95 | B2 mAP@50 / @50-95 | B3 mAP@50 / @50-95 |
|---|---:|---:|---:|
| SwinIR-upsampled images（SwinIR 放大的影像） | 34.23% / 13.80% | 33.65% / 13.52% | 42.79% / 18.50% |
| NGram-MoSE-upsampled images（NGram-MoSE 放大的影像） | 39.65% / 16.38% | 38.66% / 16.04% | 43.41% / 18.34% |

In the paper, the B3 mAP@50-95 of the NGram-MoSE-trained system (18.34%) is set in bold, but the SwinIR-trained system scores higher on that metric (18.50%). The mAP values in Tables III and IV are box metrics of the YOLOv8s-seg detector, not mask metrics.

論文中 B3 的 mAP@50-95 把 NGram-MoSE 的 18.34% 標成粗體，但這個指標其實是 SwinIR 的 18.50% 比較高。Table III、IV 的 mAP 都是 YOLOv8s-seg 的 box metrics，不是 mask metrics。

## Repository structure Repo 結構

```
models/
  network_fusion.py      NGswin_MoSE network and NSTB_MoSE block       網路與 NSTB_MoSE block
  ngram_ops.py           N-Gram context, window attention (RPE + LePE)
  moe.py                 MoE FFN, per-window dispatcher, Smart Merger
  network_swinir.py      SwinIR baseline
utils/
  losses.py              L1 + SSIM + NCC + VGG perceptual + MoE balance loss
  checkpoint.py          checkpoint loader (fails on missing/unexpected keys)   key 對不上會直接報錯
data/dataset.py          GeoTIFF/PNG/NPY datasets (random-crop train, full-image val)
data_scripts/            Sentinel-2 training-scene download and tiling   下載訓練場景並切 tile
train.py                 training (NGram-MoSE or SwinIR)                 訓練
benchmark_complexity.py  Table II
download_40_ood_testset.py, generate_paper_ood_results.py   Table I
generate_top3_paper_figures.py                              Fig. 4
run_inference.py         SR for arbitrary LR GeoTIFF/PNG files           對任意 LR 影像做 SR
check_leakage.py         checks that train/val/test tiles do not overlap by file name   依檔名檢查切分有沒有重疊
landslide/               Landslide4Sense downstream evaluation (Tables III, IV)        下游評估
```

## Installation 安裝

```bash
conda create -n ngram_mose python=3.10 -y
conda activate ngram_mose
# Install PyTorch for your CUDA version first / 先依 CUDA 版本安裝 PyTorch: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
# ultralytics depends on ultralytics-thop, which installs into the same `thop` module and counts
# FLOPs differently. Reinstall the original thop so that Table II is reproduced exactly.
# ultralytics 會連帶裝 ultralytics-thop，跟原版 thop 共用同一個模組、FLOPs 算法也不同，
# 要重裝原版 thop，Table II 才會完全一致。
pip install --force-reinstall --no-deps thop==0.1.1.post2209072238
```

If the `defaults` channel asks you to accept Anaconda's Terms of Service, you can create the environment from conda-forge instead: `conda create -n ngram_mose -c conda-forge --override-channels python=3.10 -y`.

如果 conda 的 `defaults` channel 要求同意 Anaconda 服務條款，可以改從 conda-forge 建環境：`conda create -n ngram_mose -c conda-forge --override-channels python=3.10 -y`。

All commands below are run from the repository root.

以下所有指令都在 repo 根目錄執行。

## Pretrained weights 預訓練權重

Download the weights from the [Releases](../../releases) page into `checkpoints/`.

到 [Releases](../../releases) 頁面下載權重，放到 `checkpoints/`。

| File | Model 模型 | SHA-256 |
|---|---|---|
| `ngram_mose_x3_e500.pth` | NGram-MoSE ×3, epoch 500 (paper 論文) | `ab450ecd8fbb31a3d3899d2863fbbbd0de93e2e77bae4f0fb4aa873cc5e8a372` |
| `swinir_x3_e90.pth` | SwinIR ×3, epoch 90 (Table I) | `3beb2cdc3b17785d2b4d1ce350b64d68c76941ea290272cb1d34e12edc3e3f91` |
| `swinir_x3_e500.pth` | SwinIR ×3, epoch 500 (final 最終權重; Tables III, IV) | `c85d5ed2943a298db195f50c6cfac66cbfc036e43543d318019e0356e849724c` |
| `l4s_yolov8s-seg_M1_native.pt` | YOLOv8s-seg detector M1, trained on native 128×128 L4S images（用原生解析度影像訓練）(Table III) | `0532c34721a0cb0368a60a33ed4d35596cf9c397f5cd694d2f85d47b23669365` |
| `l4s_yolov8s-seg_M2_ngram-mose-sr.pt` | Detector M2, trained on NGram-MoSE-upsampled 384×384 images（用 NGram-MoSE 放大的影像訓練）(Table IV) | `504453f420efbfd167b34436ca3e86d079bb5f751764ff3a40f4c084b9eb85be` |
| `l4s_yolov8s-seg_M2_swinir-sr.pt` | Detector M2, trained on SwinIR-upsampled 384×384 images（用 SwinIR 放大的影像訓練）(Table IV) | `bf62a68a3b3812a2f234477db32127a3a4c0e33e5f6ed5bbf5b0ee1b72ade2bf` |

The SR checkpoints are the original training checkpoints with the `total_ops` / `total_params` buffers removed. `thop` adds these buffers when `train.py` profiles the model. The weights are unchanged.

SR 權重是原始訓練的 checkpoint，只移除了 `total_ops` / `total_params` 這兩種 buffer，權重本身沒有改。這些 buffer 是 `train.py` 用 `thop` profile 模型時加進去的。

**Which SwinIR checkpoint each table uses.** The SwinIR row of Table I was produced with `swinir_x3_e90.pth`. The original evaluation script loaded the checkpoint with the newest file-modification time, and after the checkpoint folder had been copied that was epoch 90 rather than the final epoch 500. Tables III and IV used the final `swinir_x3_e500.pth`. With the epoch-500 checkpoint, SwinIR scores 31.41 dB / 0.9034 on the Table I test set, still below NGram-MoSE (31.68 dB / 0.9089). Both checkpoints are released. `generate_paper_ood_results.py` defaults to the epoch-90 checkpoint so that it reproduces Table I; pass `--ckpt_swinir checkpoints/swinir_x3_e500.pth` to evaluate the final one.

**每張表用的是哪個 SwinIR checkpoint：** Table I 的 SwinIR 用的是 `swinir_x3_e90.pth`。原本的評估腳本會載入「檔案修改時間最新」的 checkpoint，而 checkpoint 資料夾被整批複製過，所以載到的是 epoch 90，不是最後的 epoch 500。Table III、IV 用的則是最終的 `swinir_x3_e500.pth`。改用 epoch 500 的話，SwinIR 在 Table I 測試集上是 31.41 dB / 0.9034，仍然低於 NGram-MoSE（31.68 dB / 0.9089）。兩個 checkpoint 都有附上。`generate_paper_ood_results.py` 預設用 epoch 90，才能重現 Table I；要評估最終權重，請加 `--ckpt_swinir checkpoints/swinir_x3_e500.pth`。

## Data 資料

### Training set (Sentinel-2 L2A, Taiwan) 訓練集

```bash
python data_scripts/download_train_data.py      # -> dataset_source_raw/
python data_scripts/prepare_self_supervised.py  # -> dataset_self_supervised/{train,val,test}_{hr,lr}
```

The downloader queries the Earth Search STAC API (`sentinel-2-l2a`, 2023, cloud cover < 5 %). For each location it takes the RGB bands (B04, B03, B02) of the least-cloudy item and crops 5120×5120 pixels around the **tile centre**. The paper used these items:

下載程式會查詢 Earth Search STAC API（`sentinel-2-l2a`、2023 年、雲量 < 5 %），每個地點取雲量最少那筆 item 的 RGB 波段（B04、B03、B02），在 **tile 中心**裁切 5120×5120 pixels。論文用的 item 如下：

| File prefix 檔名前綴 | Sentinel-2 item |
|---|---|
| `Taipei_City` | `S2A_51RUH_20230306_0_L2A` |
| `Taichung_City` | `S2B_50QRM_20230130_0_L2A` |
| `Central_Mountain` | `S2B_50QRM_20230130_0_L2A` |
| `Tainan_City` | `S2B_51QTF_20230301_0_L2A` |
| `Kaohsiung_Port` | `S2B_51QTF_20230301_0_L2A` |

**Only 3 of the 5 scenes are unique.** The downloader crops around the tile centre, not around the location coordinates. `Taichung_City` and `Central_Mountain` therefore resolve to byte-identical crops of tile 50QRM, and `Tainan_City` and `Kaohsiung_Port` to byte-identical crops of tile 51QTF. The paper's training set contains 3 unique scenes, not the 5 locations the paper lists. The downloader is kept unchanged so that it reproduces the paper's training data.

**5 個場景中只有 3 個不重複：** 下載程式以 tile 中心裁切，不是以指定座標為中心。所以 `Taichung_City` 和 `Central_Mountain` 裁出的是 tile 50QRM 中完全相同的區域（byte-identical），`Tainan_City` 和 `Kaohsiung_Port` 也是 tile 51QTF 中完全相同的區域。論文的訓練集實際上只有 3 個不重複的場景，不是論文列出的 5 個地點。為了能重現論文的訓練資料，下載程式維持原樣。

`prepare_self_supervised.py` cuts each scene into non-overlapping 480×480 HR tiles. It synthesises the LR tiles with a Gaussian blur (σ = 1) followed by ×1/3 bicubic downsampling, then assigns tiles at random to train/val/test (80/10/10). The split is not seeded; the paper run had 408/42/50 tiles.

`prepare_self_supervised.py` 會把每個場景切成不重疊的 480×480 HR tile，對 HR 做 Gaussian blur（σ = 1）再 ×1/3 bicubic 縮小產生 LR，然後隨機分成 train/val/test（80/10/10）。這個切分沒有固定 seed，論文那次是 408/42/50 個 tile。

### OOD test set (Table I) OOD 測試集

```bash
python download_40_ood_testset.py   # -> dataset_40_OOD_testset/
```

This script downloads 1200×1200 RGB windows around 40 city centres outside Taiwan from Microsoft Planetary Computer (2023, cloud cover < 5 %). Windows that touch a tile edge or contain more than 1 % no-data pixels are skipped. In the paper run, 35 locations downloaded successfully; Singapore, Kuala Lumpur, Auckland, São Paulo and Johannesburg failed. `generate_paper_ood_results.py` then excludes 6 scenes with thin cloud or haze (Berlin, Rio de Janeiro, Athens, Congo Basin, San Francisco, Los Angeles). This leaves the 29 scenes of Table I. The catalog search is not pinned to item IDs, so a fresh download may return different acquisitions.

這支腳本會從 Microsoft Planetary Computer 下載台灣以外 40 個城市中心周圍的 1200×1200 RGB 影像（2023 年、雲量 < 5 %），碰到 tile 邊緣或 no-data pixel 超過 1 % 的會跳過。論文那次有 35 個地點下載成功，Singapore、Kuala Lumpur、Auckland、São Paulo、Johannesburg 失敗。`generate_paper_ood_results.py` 再排除 6 個有薄雲或霧霾的場景（Berlin、Rio de Janeiro、Athens、Congo Basin、San Francisco、Los Angeles），剩下的就是 Table I 的 29 個場景。搜尋時沒有固定 item ID，所以重新下載可能會拿到不同日期的影像。

### Landslide4Sense

Download the Landslide4Sense training data (`img/*.h5`, `mask/*.h5`) from the [official repository](https://github.com/iarai/Landslide4Sense-2022) into `raw_data/TrainData/`.

從 [官方 repo](https://github.com/iarai/Landslide4Sense-2022) 下載 Landslide4Sense 訓練資料（`img/*.h5`、`mask/*.h5`），放到 `raw_data/TrainData/`。

## Training 訓練

```bash
# NGram-MoSE (paper configuration / 論文設定)
WANDB_MODE=disabled python train.py --model_type ngram --use_moe --experts 8 --epochs 500 --batch_size 4 --run_name NGram_Sentinel_Run

# SwinIR baseline (LR patch 64x64; the original batch size was not recorded / 原本的 batch size 沒有紀錄)
WANDB_MODE=disabled python train.py --model_type swinir --patch_size 64 --epochs 500 --run_name SwinIR_Run
```

Training settings:

- Defaults: scale ×3, LR patch 128×128, batch size 16, AdamW (β = 0.9, 0.999), learning rate 1e-4, seed 100.
- The paper's NGram-MoSE run used batch size 4 instead of the default 16, according to its Weights & Biases log (500 epochs, about 31 hours).
- Loss weights: L1 1.0, SSIM 1.0, NCC 0.5, VGG-19 `relu5_4` perceptual 0.1, MoE load balancing 0.2.
- `--use_moe` is required for NGram-MoSE. Without it, the blocks use a dense MLP.
- Checkpoints are written to `checkpoints/<run_name>/model_epoch_<N>.pth` every 10 epochs.
- Metrics are logged to Weights & Biases. Use `WANDB_MODE=disabled` (or `offline`) to train without an account.

訓練設定：

- 預設值：scale ×3、LR patch 128×128、batch size 16、AdamW（β = 0.9, 0.999）、learning rate 1e-4、seed 100。
- 根據 Weights & Biases 紀錄，論文的 NGram-MoSE 是用 batch size 4 訓練的（500 epochs，約 31 小時），不是預設的 16。
- Loss weights：L1 1.0、SSIM 1.0、NCC 0.5、VGG-19 `relu5_4` perceptual 0.1、MoE load balancing 0.2。
- NGram-MoSE 一定要加 `--use_moe`，不加的話 block 會改用 dense MLP。
- 每 10 個 epoch 存一次 checkpoint，路徑是 `checkpoints/<run_name>/model_epoch_<N>.pth`。
- 訓練過程會記錄到 Weights & Biases；沒有帳號的話，加上 `WANDB_MODE=disabled`（或 `offline`）即可。

**SwinIR patch size.** The released SwinIR checkpoints were trained with 64×64 LR patches; their attention-mask shape shows this. NGram-MoSE was trained with 128×128 patches. The batch size of the SwinIR run was not recorded. If the default batch size of 16 runs out of GPU memory, lower `--batch_size`.

**SwinIR 的 patch 大小：** 從 attention mask 的形狀可以看出，附上的 SwinIR checkpoint 是用 64×64 LR patch 訓練的，NGram-MoSE 則是 128×128。SwinIR 那次訓練的 batch size 沒有紀錄。如果預設的 batch size 16 讓 GPU 記憶體不夠，請調小 `--batch_size`。

## Evaluation 評估

```bash
python benchmark_complexity.py         # Table II (no weights or data needed / 不需要權重或資料)
python generate_paper_ood_results.py   # Table I + comparison strips / 比較圖 -> dataset_40_OOD_testset_results/
python generate_top3_paper_figures.py  # Fig. 4: top-3 OOD scenes by PSNR gain over bicubic / PSNR 進步最多的 3 個場景
python run_inference.py --input path/to/lr.tif --output sr.png
```

Evaluation protocol of `generate_paper_ood_results.py`:

- Crop the central 480×480 pixels and scale to reflectance (DN / 10000). If the 1st percentile exceeds 800 DN, subtract 1000 DN first: Sentinel-2 processing baseline 04.00 added this offset from 2022 onwards.
- Generate the LR input with ×1/3 `cv2.INTER_AREA` downsampling. This is a different degradation from the one used for training.
- Compute PSNR and SSIM (`piq`) on RGB reflectance clipped to [0, 1].

`generate_paper_ood_results.py` 的評估流程：

- 裁出中央 480×480 pixels，換算成 reflectance（DN / 10000）。如果第 1 百分位數超過 800 DN，先減掉 1000 DN；這是 Sentinel-2 從 2022 年 processing baseline 04.00 開始加上的 offset。
- 用 ×1/3 `cv2.INTER_AREA` 縮小產生 LR 輸入。注意這跟訓練時的降質方式不同。
- 在裁切到 [0, 1] 的 RGB reflectance 上計算 PSNR 和 SSIM（`piq`）。

Running the steps above with the released weights reproduces Tables I and II exactly. The SSIM of SwinIR is 0.8945, against 0.8946 in the paper.

用附上的權重照上面步驟跑，可以完全重現 Table I、II。唯一的差異是 SwinIR 的 SSIM：重跑是 0.8945，論文是 0.8946。

## Landslide4Sense downstream evaluation (Tables III, IV) 下游評估

```bash
# 1. Split the L4S training set 80/10/10 (seed 42) and build the image sets and YOLO polygon labels
#    把 L4S 訓練集切成 80/10/10，產生各組影像和 YOLO polygon label
python landslide/prepare_experiment.py

# 2. Super-resolve with both models: 128x128 -> 384x384 (Table IV) and 42x42 -> 126x126 (Table III)
#    兩個模型都做 SR
for split in raw_h5 raw_h5_lr; do
  python landslide/run_inference_l4s_fair.py --input landslide_workdir/$split --output SR_temp_fair_ngram  --model_type ngram  --ckpt_dir checkpoints/ngram_mose_x3_e500.pth
  python landslide/run_inference_l4s_fair.py --input landslide_workdir/$split --output SR_temp_fair_swinir --model_type swinir --ckpt_dir checkpoints/swinir_x3_e500.pth
done

# 3. NGram-MoSE: move its SR images into the YOLO dataset folders, train M1 and M2, evaluate all scenarios
#    把 NGram-MoSE 的 SR 影像搬進 YOLO 資料夾，訓練 M1、M2，評估所有 scenario
python landslide/finalize_sr.py --src SR_temp_fair_ngram
python landslide/train_manage.py --train M1   # native 128x128 images / 原生解析度影像
python landslide/train_manage.py --train M2   # NGram-MoSE-upsampled 384x384 images / NGram-MoSE 放大的影像
for s in A1 A2 A3 A4 B1 B2 B3; do python landslide/train_manage.py --eval $s; done

# 4. SwinIR: same, with the SwinIR SR images. M1 does not need to be retrained.
#    SwinIR 同樣流程，M1 不用重新訓練。--train M2 會覆蓋 NGram-MoSE 的 M2，需要的話請先備份。
python landslide/finalize_sr.py --src SR_temp_fair_swinir
python landslide/train_manage.py --train M2
for s in A3 B1 B2 B3; do python landslide/train_manage.py --eval $s; done
```

Results are appended to `experiment_results.txt`. `--eval` also writes the test images with the predicted masks and boxes to `runs/detect/<scenario>/`; these were used for Figs. 5 and 6. To evaluate the released detectors without training, pass their paths:

結果會附加到 `experiment_results.txt`。`--eval` 也會把畫上預測 mask 和 box 的測試影像輸出到 `runs/detect/<scenario>/`，論文的 Fig. 5、6 就是用這些圖。不想重新訓練的話，可以直接指定 Release 的偵測器權重：

```bash
python landslide/train_manage.py --eval A3 --m1 checkpoints/l4s_yolov8s-seg_M1_native.pt --m2 checkpoints/l4s_yolov8s-seg_M2_ngram-mose-sr.pt
```

| Paper 論文 | Scenario | Detector 偵測器 | Test images 測試影像 |
|---|---|---|---|
| Table III: low-resolution input | `A1` | M1 | 42×42 |
| Table III: bicubic | `A2` | M1 | 42 → 128 bicubic |
| Table III: SR | `A3` | M1 | 42 → 126 SR |
| Table III: native reference | `A4` | M1 | 128×128 native 原生解析度 |
| Table IV: B1 native input | `B1` | M2 | 128×128 native 原生解析度 |
| Table IV: B2 bicubic | `B2` | M2 | 128 → 384 bicubic |
| Table IV: B3 SR | `B3` | M2 | 128 → 384 SR |

Notes on the paper run:

- The reported mAP values are YOLOv8s-seg **box** metrics (`metrics.box.map50`, `metrics.box.map`).
- Both SR models were run with `run_inference_l4s_fair.py`, with no pre-blur and no test-time augmentation, and the same fixed visualisation (clip 0.15, gamma 1.2).
- The detectors were trained with Ultralytics **8.4.18** (seed 0, `deterministic=True`, 200 epochs with early-stopping patience 50; training stopped at epochs 160 and 125). Re-running the evaluation with the released detectors and the original test images reproduces all 14 scenario results within 0.06 percentage points. Newer Ultralytics versions compute mAP slightly differently (by about 0.1–0.3 points), so `requirements.txt` pins 8.4.18.

論文那次實驗的補充說明：

- 回報的 mAP 是 YOLOv8s-seg 的 **box** metrics（`metrics.box.map50`、`metrics.box.map`）。
- 兩個 SR 模型都用 `run_inference_l4s_fair.py` 推論：不做 pre-blur、不做 test-time augmentation，視覺化方式也相同（clip 0.15、gamma 1.2）。
- 偵測器是用 Ultralytics **8.4.18** 訓練的：seed 0、`deterministic=True`、200 epochs、early-stopping patience 50，實際分別在 epoch 160 和 125 停止。用附上的偵測器和原始測試影像重跑，14 個 scenario 的結果都在 0.06 個百分點以內。新版 Ultralytics 算 mAP 的方式略有不同（差約 0.1–0.3 個百分點），所以 `requirements.txt` 固定使用 8.4.18。

## Implementation notes 實作細節

- **Parameter count.** `thop` reports 1.70 M parameters, the value in Table II. Counting all `nn.Parameter` tensors gives 1.72 M; the difference is the MoE gate and the relative-position-bias tables, which `thop` does not count. `benchmark_complexity.py` prints both numbers.
- **FLOPs counter.** Table II was computed with `thop` 0.1.1. `ultralytics-thop` 2.2.2 counts more operations and reports 15.83 G (NGram-MoSE) and 208.93 G (SwinIR), a 13.2× reduction. `benchmark_complexity.py` warns when a different `thop` is installed.
- **Attention heads.** The first stage is configured with 6 heads. 64 channels do not divide by 6, so the code reduces it to 4 heads and prints a `[Warning] Dim 64 cannot be divided by heads 6` message. This is expected.
- **MoE routing.** Routing is computed per 8×8 window, from mean-pooled window features: all tokens of a window go to the same top-2 experts.

中文說明：

- **參數量：** `thop` 算出 1.70 M，也就是 Table II 的數字。把所有 `nn.Parameter` 都算進去是 1.72 M，差在 MoE gate 和 relative-position-bias table，這兩個 `thop` 不會算。`benchmark_complexity.py` 會兩個數字都印出來。
- **FLOPs 計算工具：** Table II 是用 `thop` 0.1.1 算的。`ultralytics-thop` 2.2.2 會多算一些運算，得到 15.83 G（NGram-MoSE）和 208.93 G（SwinIR），也就是 13.2×。裝到不同版本的 `thop` 時，`benchmark_complexity.py` 會發出警告。
- **Attention heads：** 第一個 stage 設定 6 個 head，但 64 個 channel 不能被 6 整除，程式會自動改成 4 個 head，並印出 `[Warning] Dim 64 cannot be divided by heads 6`。這是正常的。
- **MoE routing：** routing 以 8×8 window 為單位，用 window 內特徵的平均值選 expert，同一個 window 的所有 token 會送到同樣的 top-2 experts。

## Citation 引用

```bibtex
@inproceedings{huang2026ngrammose,
  title     = {NGram-MoSE: Efficient Remote Sensing Super-Resolution via N-Gram Context and Mixture-of-Experts},
  author    = {Huang, Yun-Hsuan and Bui, Trong-An and Chuang, Chih-Hung},
  booktitle = {2026 IEEE 13th International Workshop on Metrology for AeroSpace (MetroAeroSpace)},
  pages     = {397--402},
  year      = {2026},
  doi       = {10.1109/METROAEROSPACE69299.2026.11646695}
}
```

## Acknowledgements 致謝

This work was partly supported by the National Science and Technology Council (NSTC), Taiwan, under Grant Nos. NSTC 114-2221-E-027-030-MY2 and 114-2218-E-027-003. The code builds on [SwinIR](https://github.com/JingyunLiang/SwinIR), [NGramSwin](https://github.com/rami0205/NGramSwin) and [Swin2-MoSE](https://github.com/IMPLabUniPr/swin2-mose). The downstream evaluation uses [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and [Landslide4Sense](https://github.com/iarai/Landslide4Sense-2022).

本研究部分由國家科學及技術委員會（NSTC）補助，計畫編號 NSTC 114-2221-E-027-030-MY2 及 114-2218-E-027-003。程式碼參考了 [SwinIR](https://github.com/JingyunLiang/SwinIR)、[NGramSwin](https://github.com/rami0205/NGramSwin) 和 [Swin2-MoSE](https://github.com/IMPLabUniPr/swin2-mose)。下游評估使用了 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) 和 [Landslide4Sense](https://github.com/iarai/Landslide4Sense-2022)。

## License 授權

*TODO:* not yet decided. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

*TODO：* 尚未決定，見 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
