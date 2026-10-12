"""Paper Table II: parameters and FLOPs for a 128x128 LR input (no weights or data needed)."""
import argparse
import warnings
import torch
import thop
from thop import profile

from models.network_fusion import NGswin_MoSE
from models.network_swinir import SwinIR
warnings.filterwarnings('ignore')


def complexity(model, size, device):
    model = model.to(device).eval()
    dummy = torch.randn(1, 3, size, size, device=device)
    with torch.no_grad():
        flops, thop_params = profile(model, inputs=(dummy,), verbose=False)
    all_params = sum(p.numel() for p in model.parameters())
    return flops / 1e9, thop_params / 1e6, all_params / 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=128, help='LR input height/width')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = {
        'SwinIR': SwinIR(upscale=3, in_chans=3, img_size=args.size, window_size=8, img_range=1.,
                         depths=[6] * 6, embed_dim=180, num_heads=[6] * 6, mlp_ratio=2,
                         upsampler='pixelshuffle', resi_connection='1conv'),
        'NGram-MoSE': NGswin_MoSE(upscale=3, use_moe=True, moe_experts=8),
    }
    # thop counts only the parameters of the layers it profiles; the paper reports that number.
    # "All params" also includes nn.Parameter tensors such as the MoE gate and position-bias tables.
    # The paper numbers were computed with thop 0.1.1; ultralytics-thop (same module name) counts differently.
    if getattr(thop, '__version__', None) != '0.1.1':
        print(f"WARNING: thop {getattr(thop, '__version__', '?')} is installed; Table II was computed with thop 0.1.1.")
    print(f"Input: {args.size}x{args.size}")
    print(f"{'Method':<12} | {'Params (M, thop)':>16} | {'All params (M)':>14} | {'FLOPs (G)':>9}")
    flops = {}
    for name, model in models.items():
        g, p_thop, p_all = complexity(model, args.size, device)
        flops[name] = g
        print(f"{name:<12} | {p_thop:>16.2f} | {p_all:>14.2f} | {g:>9.2f}")
    print(f"FLOPs reduction: {flops['SwinIR'] / flops['NGram-MoSE']:.1f}x")


if __name__ == "__main__":
    main()
