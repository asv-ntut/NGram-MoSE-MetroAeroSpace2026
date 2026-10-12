# NGram-MoSE

Code for **NGram-MoSE: Efficient Remote Sensing Super-Resolution via N-Gram Context and Mixture-of-Experts**, 2026 IEEE 13th International Workshop on Metrology for AeroSpace (MetroAeroSpace), pp. 397–402.

Yun-Hsuan Huang, Trong-An Bui, Chih-Hung Chuang
Institute of Aerospace and Systems Engineering, National Taipei University of Technology

[[Paper (IEEE Xplore)]](https://ieeexplore.ieee.org/document/11646695) · DOI [10.1109/METROAEROSPACE69299.2026.11646695](https://doi.org/10.1109/METROAEROSPACE69299.2026.11646695)

This repository is the code release for the workshop paper (git tag `v1.0-metroaerospace2026`).

NGram-MoSE is a lightweight Transformer for ×3 super-resolution of Sentinel-2 RGB imagery. Each N-Gram Swin Transformer Block (NSTB) combines:

- **N-Gram context injection** (n = 2): forward and backward sliding-window attention over neighbouring windows, added to the window tokens before window self-attention. This improves continuity across window boundaries.
- **A sparse Mixture-of-Experts FFN**: 8 experts with top-2 gating and a load-balancing loss, followed by a **Smart Merger** (3×3 depth-wise convolution + residual) that restores local spatial coherence.

The network has 14 NSTB blocks (stages of 6/4/4), embedding dimension 64, window size 8, and a pixel-shuffle reconstruction head.

> **TODO before making this repository public:** resolve the items marked *TODO* below, choose a license (see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)), then delete this note.

## Results

**Table I: ×3 restoration on the out-of-distribution (OOD) test set** (29 scenes, 480×480 HR)

| Method | PSNR (dB) | SSIM |
|---|---:|---:|
| Bicubic | 31.43 | 0.8873 |
| SwinIR | 31.31 | 0.8946 |
| **NGram-MoSE** | **31.68** | **0.9089** |

**Table II: complexity for a 128×128 LR input**

| Method | Params (M) | FLOPs (G) |
|---|---:|---:|
| SwinIR | 11.89 | 195.34 |
| **NGram-MoSE** | **1.70** | **13.94** |

**Table III: Landslide4Sense, detector trained on native-resolution imagery** (YOLOv8s-seg)

| Input | mAP@50 | mAP@50-95 |
|---|---:|---:|
| Low-resolution input | 23.03% | 8.01%* |
| Bicubic | 25.64% | 9.49% |
| SwinIR | 25.61% | 8.47% |
| **NGram-MoSE** | **30.11%** | **10.49%** |
| Native-resolution reference | 43.88% | 19.26% |

\* *TODO:* The original evaluation log and the re-run both give 8.28%; the 8.01% in the paper looks like a transcription error.

**Table IV: cross-scale robustness** (detector M2 trained on SR-upsampled 384×384 images; B1 native input, B2 bicubic-upsampled input, B3 SR-upsampled input)

| Detector trained on | B1 mAP@50 / @50-95 | B2 mAP@50 / @50-95 | B3 mAP@50 / @50-95 |
|---|---:|---:|---:|
| SwinIR-upsampled images | 34.23% / 13.80% | 33.65% / 13.52% | 42.79% / 18.50% |
| NGram-MoSE-upsampled images | 39.65% / 16.38% | 38.66% / 16.04% | 43.41% / 18.34% |

## Repository structure

```
models/
  network_fusion.py      NGswin_MoSE network and NSTB_MoSE block
  ngram_ops.py           N-Gram context, window attention (RPE + LePE)
  moe.py                 MoE FFN, per-window dispatcher, Smart Merger
  network_swinir.py      SwinIR baseline
utils/
  losses.py              L1 + SSIM + NCC + VGG perceptual + MoE balance loss
  checkpoint.py          checkpoint loader (fails on missing/unexpected keys)
data/dataset.py          GeoTIFF/PNG/NPY datasets (random-crop train, full-image val)
data_scripts/            Sentinel-2 training-scene download and tiling
train.py                 training (NGram-MoSE or SwinIR)
benchmark_complexity.py  Table II
download_40_ood_testset.py, generate_paper_ood_results.py   Table I
generate_top3_paper_figures.py                              Fig. 4 panels
run_inference.py         SR for arbitrary LR GeoTIFF/PNG files
check_leakage.py         checks that train/val/test tiles do not overlap by file name
landslide/               Landslide4Sense downstream evaluation (Tables III, IV)
```

## Installation

```bash
conda create -n ngram_mose python=3.10 -y
conda activate ngram_mose
# Install PyTorch for your CUDA version first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
# ultralytics depends on ultralytics-thop, which installs into the same `thop` module and counts
# FLOPs differently. Reinstall the original thop so that Table II is reproduced exactly:
pip install --force-reinstall --no-deps thop==0.1.1.post2209072238
```

If the `defaults` channel asks you to accept Anaconda's Terms of Service, you can create the environment from conda-forge instead: `conda create -n ngram_mose -c conda-forge --override-channels python=3.10 -y`.

All commands below are run from the repository root.

## Pretrained weights

Download the weights from the [Releases](../../releases) page into `checkpoints/`:

| File | Model | SHA-256 |
|---|---|---|
| `ngram_mose_x3_e500.pth` | NGram-MoSE ×3, epoch 500 (paper) | `ab450ecd8fbb31a3d3899d2863fbbbd0de93e2e77bae4f0fb4aa873cc5e8a372` |
| `swinir_x3_e90.pth` | SwinIR ×3, epoch 90 (Table I) | `3beb2cdc3b17785d2b4d1ce350b64d68c76941ea290272cb1d34e12edc3e3f91` |
| `swinir_x3_e500.pth` | SwinIR ×3, epoch 500 (final; Tables III and IV) | `c85d5ed2943a298db195f50c6cfac66cbfc036e43543d318019e0356e849724c` |
| `l4s_yolov8s-seg_M1_native.pt` | YOLOv8s-seg detector M1, trained on native 128×128 L4S images (Table III) | `0532c34721a0cb0368a60a33ed4d35596cf9c397f5cd694d2f85d47b23669365` |
| `l4s_yolov8s-seg_M2_ngram-mose-sr.pt` | Detector M2, trained on NGram-MoSE-upsampled 384×384 images (Table IV) | `504453f420efbfd167b34436ca3e86d079bb5f751764ff3a40f4c084b9eb85be` |
| `l4s_yolov8s-seg_M2_swinir-sr.pt` | Detector M2, trained on SwinIR-upsampled 384×384 images (Table IV) | `bf62a68a3b3812a2f234477db32127a3a4c0e33e5f6ed5bbf5b0ee1b72ade2bf` |

The released files are the original training checkpoints with the `total_ops` / `total_params` buffers removed. `thop` adds these buffers when `train.py` profiles the model. The weights are unchanged.

> *TODO:* The SwinIR row of Table I was produced with the epoch-90 checkpoint, while Tables III and IV used the final epoch-500 checkpoint. With the epoch-500 checkpoint, SwinIR scores 31.41 dB / 0.9034 in Table I, still below NGram-MoSE. Decide which checkpoint the repository presents as the baseline, and whether the paper needs an erratum.

## Data

### Training set (Sentinel-2 L2A, Taiwan)

```bash
python data_scripts/download_train_data.py      # -> dataset_source_raw/
python data_scripts/prepare_self_supervised.py  # -> dataset_self_supervised/{train,val,test}_{hr,lr}
```

The downloader queries the Earth Search STAC API (`sentinel-2-l2a`, 2023, cloud cover < 5 %). For each location it takes the RGB bands (B04, B03, B02) of the least-cloudy item and crops 5120×5120 pixels around the **tile centre**. The paper used these items:

| File prefix | Sentinel-2 item |
|---|---|
| `Taipei_City` | `S2A_51RUH_20230306_0_L2A` |
| `Taichung_City` | `S2B_50QRM_20230130_0_L2A` |
| `Central_Mountain` | `S2B_50QRM_20230130_0_L2A` |
| `Tainan_City` | `S2B_51QTF_20230301_0_L2A` |
| `Kaohsiung_Port` | `S2B_51QTF_20230301_0_L2A` |

> *TODO:* The crop is taken at the tile centre, not at the location coordinates. Two pairs of locations therefore resolve to byte-identical crops, so the training data contains 3 unique scenes, not 5. Decide how to describe this, and whether to fix the downloader for future use.

`prepare_self_supervised.py` cuts each scene into non-overlapping 480×480 HR tiles. It synthesises the LR tiles with a Gaussian blur (σ = 1) followed by ×1/3 bicubic downsampling, then assigns tiles at random to train/val/test (80/10/10). The split is not seeded; the paper run had 408/42/50 tiles.

### OOD test set (Table I)

```bash
python download_40_ood_testset.py   # -> dataset_40_OOD_testset/
```

This script downloads 1200×1200 RGB windows around 40 city centres outside Taiwan from Microsoft Planetary Computer (2023, cloud cover < 5 %). Windows that touch a tile edge or contain more than 1 % no-data pixels are skipped. In the paper run, 35 locations downloaded successfully; Singapore, Kuala Lumpur, Auckland, São Paulo and Johannesburg failed. `generate_paper_ood_results.py` then excludes 6 scenes with thin cloud or haze (Berlin, Rio de Janeiro, Athens, Congo Basin, San Francisco, Los Angeles). This leaves the 29 scenes of Table I.

The catalog search is not pinned to item IDs, so a fresh download may return different acquisitions.

### Landslide4Sense

Download the Landslide4Sense training data (`img/*.h5`, `mask/*.h5`) from the [official repository](https://github.com/iarai/Landslide4Sense-2022) into `raw_data/TrainData/`.

## Training

```bash
# NGram-MoSE (paper configuration)
WANDB_MODE=disabled python train.py --model_type ngram --use_moe --experts 8 --epochs 500 --run_name NGram_Sentinel_Run

# SwinIR baseline
WANDB_MODE=disabled python train.py --model_type swinir --patch_size 64 --batch_size 2 --epochs 500 --run_name SwinIR_Run
```

- Defaults: scale ×3, LR patch 128×128, batch size 16, AdamW (β = 0.9, 0.999), learning rate 1e-4, seed 100.
- Loss weights: L1 1.0, SSIM 1.0, NCC 0.5, VGG-19 `relu5_4` perceptual 0.1, MoE load balancing 0.2.
- `--use_moe` is required for NGram-MoSE. Without it, the blocks use a dense MLP.
- Checkpoints are written to `checkpoints/<run_name>/model_epoch_<N>.pth` every 10 epochs.
- Metrics are logged to Weights & Biases. Use `WANDB_MODE=disabled` (or `offline`) to train without an account.

> *TODO:* The released SwinIR checkpoints were trained with 64×64 LR patches (inferred from their attention-mask shape), while NGram-MoSE used 128×128. Confirm the SwinIR batch size, and decide how to describe the difference in patch size.

## Evaluation

```bash
python benchmark_complexity.py         # Table II (no weights or data needed)
python generate_paper_ood_results.py   # Table I, plus comparison strips in dataset_40_OOD_testset_results/
python generate_top3_paper_figures.py  # Fig. 4 panels: the 3 OOD scenes with the largest PSNR gain over bicubic
python run_inference.py --input path/to/lr.tif --output sr.png
```

Evaluation protocol of `generate_paper_ood_results.py`:

- Crop the central 480×480 pixels and scale to reflectance (DN / 10000). If the 1st percentile exceeds 800 DN, subtract 1000 DN first: Sentinel-2 processing baseline 04.00 added this offset from 2022 onwards.
- Generate the LR input with ×1/3 `cv2.INTER_AREA` downsampling. This is a different degradation from the one used for training.
- Compute PSNR and SSIM (`piq`) on RGB reflectance clipped to [0, 1].

Running the steps above with the released weights reproduces Tables I and II exactly. The SSIM of SwinIR is 0.8945, against 0.8946 in the paper.

## Landslide4Sense downstream evaluation (Tables III and IV)

```bash
# 1. Split the L4S training set 80/10/10 (seed 42) and build the image sets and YOLO polygon labels
python landslide/prepare_experiment.py

# 2. Super-resolve with both models: 128x128 -> 384x384 (Table IV) and 42x42 -> 126x126 (Table III)
for split in raw_h5 raw_h5_lr; do
  python landslide/run_inference_l4s_fair.py --input landslide_workdir/$split --output SR_temp_fair_ngram  --model_type ngram  --ckpt_dir checkpoints/ngram_mose_x3_e500.pth
  python landslide/run_inference_l4s_fair.py --input landslide_workdir/$split --output SR_temp_fair_swinir --model_type swinir --ckpt_dir checkpoints/swinir_x3_e500.pth
done

# 3. NGram-MoSE: move its SR images into the YOLO dataset folders, train M1 and M2, evaluate all scenarios
python landslide/finalize_sr.py --src SR_temp_fair_ngram
python landslide/train_manage.py --train M1   # trained on native 128x128 images
python landslide/train_manage.py --train M2   # trained on NGram-MoSE-upsampled 384x384 images
for s in A1 A2 A3 A4 B1 B2 B3; do python landslide/train_manage.py --eval $s; done

# 4. SwinIR: same, with the SwinIR SR images. M1 does not need to be retrained.
#    (--train M2 overwrites the NGram-MoSE M2 in runs/; back it up first if you need it.)
python landslide/finalize_sr.py --src SR_temp_fair_swinir
python landslide/train_manage.py --train M2
for s in A3 B1 B2 B3; do python landslide/train_manage.py --eval $s; done
```

Results are appended to `experiment_results.txt`. `--eval` also writes the test images with the predicted masks and boxes to `runs/detect/<scenario>/`; these were used for Figs. 5 and 6.

To evaluate the released detectors without training, pass their paths:

```bash
python landslide/train_manage.py --eval A3 --m1 checkpoints/l4s_yolov8s-seg_M1_native.pt --m2 checkpoints/l4s_yolov8s-seg_M2_ngram-mose-sr.pt
```

| Paper | Scenario | Detector | Test images |
|---|---|---|---|
| Table III: low-resolution input | `A1` | M1 | 42×42 |
| Table III: bicubic | `A2` | M1 | 42 → 128 bicubic |
| Table III: SR | `A3` | M1 | 42 → 126 SR |
| Table III: native reference | `A4` | M1 | 128×128 native |
| Table IV: B1 native input | `B1` | M2 | 128×128 native |
| Table IV: B2 bicubic | `B2` | M2 | 128 → 384 bicubic |
| Table IV: B3 SR | `B3` | M2 | 128 → 384 SR |

Notes on the paper run:

- The reported mAP values are YOLOv8s-seg **box** metrics (`metrics.box.map50`, `metrics.box.map`).
- Both SR models were run with `run_inference_l4s_fair.py`, with no pre-blur and no test-time augmentation, and the same fixed visualisation (clip 0.15, gamma 1.2).
- The detectors were trained with Ultralytics **8.4.18** (seed 0, `deterministic=True`, 200 epochs with early-stopping patience 50; training stopped at epochs 160 and 125). Re-running the evaluation with the released detectors and the original test images reproduces all 14 scenario results within 0.06 percentage points. Newer Ultralytics versions compute mAP slightly differently (by about 0.1–0.3 points), so `requirements.txt` pins 8.4.18.

## Implementation notes

- **Parameter count.** `thop` reports 1.70 M parameters, the value in Table II. Counting all `nn.Parameter` tensors gives 1.72 M; the difference is the MoE gate and the relative-position-bias tables, which `thop` does not count. `benchmark_complexity.py` prints both numbers.
- **FLOPs counter.** Table II was computed with `thop` 0.1.1. `ultralytics-thop` 2.2.2 counts more operations and reports 15.83 G (NGram-MoSE) and 208.93 G (SwinIR), a 13.2× reduction. `benchmark_complexity.py` warns when a different `thop` is installed.
- **Attention heads.** The first stage is configured with 6 heads. 64 channels do not divide by 6, so the code reduces it to 4 heads and prints a `[Warning] Dim 64 cannot be divided by heads 6` message. This is expected.
- **MoE routing.** Routing is computed per 8×8 window, from mean-pooled window features: all tokens of a window go to the same top-2 experts.

## Citation

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

## Acknowledgements

This work was partly supported by the National Science and Technology Council (NSTC), Taiwan, under Grant Nos. NSTC 114-2221-E-027-030-MY2 and 114-2218-E-027-003. The code builds on [SwinIR](https://github.com/JingyunLiang/SwinIR), [NGramSwin](https://github.com/rami0205/NGramSwin) and [Swin2-MoSE](https://github.com/IMPLabUniPr/swin2-mose). The downstream evaluation uses [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and [Landslide4Sense](https://github.com/iarai/Landslide4Sense-2022).

## License

*TODO:* not yet decided. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
