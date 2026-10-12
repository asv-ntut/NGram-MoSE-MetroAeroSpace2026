import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import to_2tuple, trunc_normal_
from torchvision.transforms import functional as TF

# --- Helper Functions ---
def window_partition(x, window_size):
    B, H, W, C = x.shape
    wh, ww = H//window_size, W//window_size
    if 0 in [wh, ww]: return x, (wh, ww)
    windows = rearrange(x, 'b (h wh) (w ww) c -> (b h w) wh ww c', wh=window_size, ww=window_size).contiguous()
    return windows, (wh, ww)

def window_unpartition(windows, num_windows):
    x = rearrange(windows, '(p h w) wh ww c -> p (h wh) (w ww) c', h=num_windows[0], w=num_windows[1])
    return x.contiguous()

# --- Fusion Attention (Swin2-MoSE: RPE + LePE) ---
class WindowAttentionFusion(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0., 
                 use_lepe=True, pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)
        self.num_heads = num_heads
        self.use_lepe = use_lepe
        
        if dim % num_heads != 0:
            new_heads = num_heads
            while dim % new_heads != 0:
                new_heads -= 1
            print(f"[Warning] Dim {dim} cannot be divided by heads {num_heads}. Adjusting heads to {new_heads}.")
            self.num_heads = new_heads
        
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        # Swin2-MoSE Paper Section 3.3: "We selected the combination of RPE and LePE"
        # So we use standard RPE (Relative Position Bias Table)
        
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        trunc_normal_(self.relative_position_bias_table, std=.02)

        # LePE (from Swin2-MoSE)
        if self.use_lepe:
            self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # RPE
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v)
        
        # LePE Calculation & Injection
        if self.use_lepe:
            H = W = int(math.sqrt(N))
            v_spatial = v.transpose(-2, -1).contiguous().view(B_, self.num_heads * self.head_dim, H, W)
            lepe = self.get_v(v_spatial)
            lepe = lepe.flatten(2).transpose(1, 2).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
            x = x + lepe

        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# --- NGram Helper Classes ---
class NGramContext(nn.Module):
    def __init__(self, dim, window_size, ngram, ngram_num_heads, padding_mode='seq_refl_win_pad'):
        super(NGramContext, self).__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)
        self.ngram = ngram
        self.padding_mode = padding_mode
        
        reduced_dim = dim // 2
        adjusted_heads = max(1, ngram_num_heads // 2)
        
        if reduced_dim % adjusted_heads != 0:
            while reduced_dim % adjusted_heads != 0:
                adjusted_heads -= 1
            if adjusted_heads < 1: adjusted_heads = 1
            
        self.unigram_embed = nn.Conv2d(dim, reduced_dim, kernel_size=self.window_size, stride=self.window_size, padding=0, groups=reduced_dim)
        
        self.ngram_attn = WindowAttentionFusion(reduced_dim, (ngram, ngram), adjusted_heads, use_lepe=False) 
        
        self.avg_pool = nn.AvgPool2d(ngram)
        self.merge = nn.Conv2d(dim, dim, 1, 1, 0)

    def seq_refl_win_pad(self, x, back=False):
        if self.ngram == 1: return x
        _, _, H, W = x.shape
        
        x_padded = TF.pad(x, (0, 0, self.ngram-1, self.ngram-1)) if not back else TF.pad(x, (self.ngram-1, self.ngram-1, 0, 0))
        
        if self.padding_mode == 'zero_pad' or H < self.ngram or W < self.ngram:
            return x_padded

        if not back:
            (start_h, start_w), (end_h, end_w) = to_2tuple(-2*self.ngram+1), to_2tuple(-self.ngram)
            x_padded[:,:,-(self.ngram-1):,:] = x_padded[:,:,start_h:end_h,:]
            x_padded[:,:,:,-(self.ngram-1):] = x_padded[:,:,:,start_w:end_w]
        else:
            (start_h, start_w), (end_h, end_w) = to_2tuple(self.ngram), to_2tuple(2*self.ngram-1)
            x_padded[:,:,:self.ngram-1,:] = x_padded[:,:,start_h:end_h,:]
            x_padded[:,:,:,:self.ngram-1] = x_padded[:,:,:,start_w:end_w]
        return x_padded
    
    def sliding_window_attention(self, unigram):
        slide = unigram.unfold(3, self.ngram, 1).unfold(2, self.ngram, 1)
        slide = rearrange(slide, 'b c h w ww hh -> b (h hh) (w ww) c')
        
        B, H, W, C = slide.shape
        wh, ww = H//self.ngram, W//self.ngram
        slide = slide.view(B, wh, self.ngram, ww, self.ngram, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.ngram*self.ngram, C)
        
        context = self.ngram_attn(slide).view(-1, self.ngram, self.ngram, C)
        context = rearrange(context, '(b h w) wh ww c -> b c (h wh) (w ww)', h=wh, w=ww)
        context = self.avg_pool(context)
        return context
        
    def forward(self, x):
        B, ph, pw, D = x.size()
        x = rearrange(x, 'b ph pw d -> b d ph pw')
        unigram = self.unigram_embed(x)
        
        unigram_forward_pad = self.seq_refl_win_pad(unigram, False)
        unigram_backward_pad = self.seq_refl_win_pad(unigram, True)
        
        context_forward = self.sliding_window_attention(unigram_forward_pad)
        context_backward = self.sliding_window_attention(unigram_backward_pad)
        
        context_bidirect = torch.cat([context_forward, context_backward], dim=1)
        context_bidirect = self.merge(context_bidirect)
        context_bidirect = rearrange(context_bidirect, 'b d h w -> b h w 1 1 d')
        
        return context_bidirect

class NGramWindowPartition(nn.Module):
    def __init__(self, dim, window_size, ngram, num_heads, shift_size=0):
        super(NGramWindowPartition, self).__init__()
        self.window_size = window_size
        self.ngram = ngram
        self.shift_size = shift_size
        self.ngram_context = NGramContext(dim, window_size, ngram, num_heads)
    
    def forward(self, x):
        B, ph, pw, D = x.size()
        context = self.ngram_context(x)
        
        windows = rearrange(x, 'b (h wh) (w ww) c -> b h w wh ww c', wh=self.window_size, ww=self.window_size).contiguous()
        
        # Out-of-place addition
        windows = windows + context
        
        if self.shift_size > 0:
            x = rearrange(windows, 'b h w wh ww c -> b (h wh) (w ww) c').contiguous()
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            windows = rearrange(shifted_x, 'b (h wh) (w ww) c -> b h w wh ww c', 
                                wh=self.window_size, ww=self.window_size).contiguous()
            
        windows = rearrange(windows, 'b h w wh ww c -> (b h w) (wh ww) c').contiguous()
        return windows, (ph//self.window_size, pw//self.window_size)