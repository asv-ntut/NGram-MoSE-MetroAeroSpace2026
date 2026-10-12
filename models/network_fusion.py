import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .moe import MoE
from .ngram_ops import NGramWindowPartition, WindowAttentionFusion, window_unpartition

# --- Helpers ---
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        
        x = self.norm(x)
        x = self.reduction(x)
        return x

# --- Fusion Block (NSTB + MoE + LePE) ---
class NSTB_MoSE(nn.Module):
    """
    Fusion Block: 
    1. N-Gram Window Partition (NGramSwin)
    2. LePE Attention (Swin2-MoSE)
    3. MoE FFN (Swin2-MoSE)
    """
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=1., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 ngram=2, use_moe=True, moe_experts=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm1 = norm_layer(dim)
        # 1. NGram Partition
        self.ngram_partition = NGramWindowPartition(dim, window_size, ngram, num_heads, shift_size)
        
        # 2. Fusion Attention (with LePE forced ON)
        self.attn = WindowAttentionFusion(dim, window_size, num_heads, qkv_bias, attn_drop, drop, use_lepe=True)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        # 3. MoE Replacement
        if use_moe:
            # Swin2-MoSE config: mlp_ratio=1, with Smart Merger
            self.mlp = MoE(dim, dim, num_experts=moe_experts, hidden_size=int(dim * mlp_ratio), 
                           k=2, with_smart_merger=True, window_size=window_size)
        else:
            self.mlp = Mlp(dim, int(dim * 2.0), act_layer=act_layer, drop=drop)

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        shortcut = x
        x = x.view(B, H, W, C)
        
        # --- Part 1: Attention ---
        # NGram Partition
        x_windows, _ = self.ngram_partition(x)
        # Attention Calculation (Fusion)
        attn_windows = self.attn(x_windows) 
        
        # Reverse Window
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_unpartition(attn_windows, (H // self.window_size, W // self.window_size))
        
        # Reverse Cyclic Shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        
        # Residual + Norm
        x = shortcut + self.drop_path(self.norm1(x))
        
        # --- Part 2: FFN / MoE ---
        loss_moe = 0
        x_norm = self.norm2(x)
        
        if isinstance(self.mlp, MoE):
            # [關鍵] 傳入 shape_info 讓 Smart Merger 能還原圖片結構進行卷積
            shape_info = (B, H, W, C)
            x_res, loss_moe = self.mlp(x_norm, shape_info=shape_info)
        else:
            x_res = self.mlp(x_norm)
            
        x = x + self.drop_path(x_res)
        return x, loss_moe

# --- Main Architecture ---
class NGswin_MoSE(nn.Module):
    """
    主架構：NGramSwin 骨架 + MoSE 組件
    """
    def __init__(self, img_size=64, in_chans=3, embed_dim=64, depths=[6,4,4], num_heads=[6,4,4],
                 window_size=8, upscale=2, use_moe=True, moe_experts=8, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.upscale = upscale
        
        # Shallow Feature Extraction
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        
        # Deep Feature Extraction (Encoder)
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            # 每個 Stage 包含多個 NSTB_MoSE Block
            layer = nn.ModuleList([
                NSTB_MoSE(embed_dim, num_heads[i_layer], window_size, 
                          shift_size=0 if (i % 2 == 0) else window_size // 2,
                          use_moe=use_moe, moe_experts=moe_experts)
                for i in range(depths[i_layer])
            ])
            self.layers.append(layer)
            
        # Reconstruction
        self.norm = nn.LayerNorm(embed_dim)
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * upscale**2, 3, 1, 1),
            nn.PixelShuffle(upscale),
            nn.Conv2d(embed_dim, in_chans, 3, 1, 1)
        )

    def forward(self, x):
        _, _, H, W = x.size()
        
        # Check Image Size & Padding
        # 確保圖片長寬是 Window Size 的倍數
        mod_pad_h = (8 - H % 8) % 8
        mod_pad_w = (8 - W % 8) % 8
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        H_pad, W_pad = x.shape[2:]
        
        # Shallow
        x = self.conv_first(x)
        x = x.flatten(2).transpose(1, 2) # [B, L, C]
        
        # Encoder (Blocks)
        total_moe_loss = 0
        for layer in self.layers:
            for block in layer:
                x, loss = block(x, (H_pad, W_pad))
                total_moe_loss += loss
        
        # Reconstruction
        x = self.norm(x)
        x = x.transpose(1, 2).view(-1, self.embed_dim, H_pad, W_pad)
        out = self.upsample(x)
        
        # Crop back to original size (remove padding)
        out = out[:, :, :H*self.upscale, :W*self.upscale]
        
        return out, total_moe_loss