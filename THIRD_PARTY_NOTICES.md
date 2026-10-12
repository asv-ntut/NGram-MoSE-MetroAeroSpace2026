# Third-party code

| Component | Used in | Upstream | Upstream license |
|---|---|---|---|
| SwinIR | `models/network_swinir.py` (copied; header retained) | [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR) | Apache-2.0 |
| NGramSwin | N-Gram context in `models/ngram_ops.py` (`NGramContext`, `NGramWindowPartition`, `seq_refl_win_pad`) | [rami0205/NGramSwin](https://github.com/rami0205/NGramSwin) | Apache-2.0 |
| Swin2-MoSE | Window attention with RPE + LePE (`models/ngram_ops.py`); MoE gating and load-balancing loss (`models/moe.py`); NCC loss (`utils/losses.py`) | [IMPLabUniPr/swin2-mose](https://github.com/IMPLabUniPr/swin2-mose) | GPL-2.0 |
| mixture-of-experts | MoE gating and `cv_squared` load-balancing loss, inherited through Swin2-MoSE | [davidmrau/mixture-of-experts](https://github.com/davidmrau/mixture-of-experts) | check upstream |
| Ultralytics YOLOv8 | dependency of `landslide/train_manage.py` (not vendored); the released `l4s_yolov8s-seg_*.pt` detectors are fine-tuned from `yolov8s-seg.pt` | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | AGPL-3.0 (check the Ultralytics terms for redistributing fine-tuned weights) |
| Landslide4Sense | training data of the released detectors | [iarai/Landslide4Sense-2022](https://github.com/iarai/Landslide4Sense-2022) | check upstream |

## Choosing this repository's license

`models/moe.py` shares its gating and load-balancing code with Swin2-MoSE, which is GPL-2.0. If that code counts as derived from Swin2-MoSE, the repository has to use a GPL-compatible license; a permissive license such as MIT would not be enough. Apache-2.0 code (SwinIR, NGramSwin) cannot be combined with GPL-2.0-only code, but it can be combined under GPL-3.0. Check the exact upstream terms (GPL-2.0-only or GPL-2.0-or-later) before you decide.

This note does not give legal advice; the authors should make the final decision.
