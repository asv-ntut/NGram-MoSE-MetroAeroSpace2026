import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .ngram_ops import window_partition, window_unpartition

class MoEExpert(nn.Module):
    def __init__(self, dim, hidden_dim, act_layer=nn.GELU, drop=0.):
        super(MoEExpert, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x: (N, L, C)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PerWindowDispatcher(object):
    """
    Per-Window Dispatcher for MoSE.
    Dispatches entire windows (L tokens) to experts based on a single gate per window.
    """
    def __init__(self, num_experts, gates, k):
        self.num_experts = num_experts
        self.gates = gates # (N_windows, num_experts)
        self.k = k
        
        # Sort experts to group assignments
        # indices: (N_windows, k)
        topk_gates, topk_indices = torch.topk(gates, k, dim=1)
        
        # Flatten to (N_windows * k) assignments
        self.batch_indices = torch.arange(gates.size(0), device=gates.device).repeat_interleave(k)
        self.expert_indices = topk_indices.flatten()
        self.gate_values = topk_gates.flatten()
        
        # Sort by expert index for efficient splitting
        sorted_experts, sort_idx = self.expert_indices.sort(0)
        self.sorted_expert_indices = sorted_experts
        self.sorted_batch_indices = self.batch_indices[sort_idx]
        self.sorted_gate_values = self.gate_values[sort_idx]
        
        # Calculate size for each expert
        # Calculate size for each expert
        # Optimized: Use bincount to avoid 8x GPU-CPU Syncs
        self.part_sizes = torch.bincount(self.sorted_expert_indices, minlength=num_experts).cpu().tolist()

    def dispatch(self, x):
        # x: (N_windows, Window_Pixels, C)
        # Reorder x based on sorted assignments
        x_exp = x[self.sorted_batch_indices] # (Total_Assignments, Window_Pixels, C)
        
        # Split for experts
        return torch.split(x_exp, self.part_sizes, dim=0)

    def combine(self, expert_out, shape_info, merger=None):
        # expert_out: list of (N_e, Window_Pixels, C)
        # shape_info: (B, H, W, C)
        
        # 1. Concatenate all expert outputs
        stitched = torch.cat(expert_out, 0) # (Total_Assignments, Window_Pixels, C)
        
        # 2. Apply Gates
        stitched = stitched * self.sorted_gate_values.view(-1, 1, 1)
        
        # 3. Add back to original windows
        # We need to sum up contributions from different experts for the same window
        B, H, W, C = shape_info
        window_size = int((stitched.shape[1])**0.5)
        num_windows = (H // window_size, W // window_size)
        total_windows = B * num_windows[0] * num_windows[1]
        
        combined_windows = torch.zeros(total_windows, window_size*window_size, C, device=stitched.device)
        combined_windows.index_add_(0, self.sorted_batch_indices, stitched)
        
        # 4. Unpartition to Full Image
        # combined_windows: (B*nH*nW, Wh*Ww, C)
        # reshape to match window_unpartition input: (B, nH, nW, Wh, Ww, C) -> flatten first 3 dims?
        # window_unpartition expects: (B*nH*nW, Wh, Ww, C)
        combined_windows = combined_windows.view(-1, window_size, window_size, C)
        combined_img = window_unpartition(combined_windows, num_windows) # (B, H, W, C)
        
        # 5. Smart Merger (Spatial Conv)
        if merger is not None:
            # (B, H, W, C) -> (B, C, H, W)
            combined_img = combined_img.permute(0, 3, 1, 2)
            combined_img = merger(combined_img)
            # (B, C, H, W) -> (B, H, W, C)
            combined_img = combined_img.permute(0, 2, 3, 1)
            
        return combined_img.flatten(1, 2) # (B, L, C)

class SmartMerger(nn.Module):
    def __init__(self, dim):
        super(SmartMerger, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        return self.conv(x) + x

class MoE(nn.Module):
    def __init__(self, input_size, output_size, num_experts, hidden_size, k=2, 
                 with_smart_merger=True, window_size=8):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size
        self.k = k
        self.with_smart_merger = with_smart_merger
        self.window_size = window_size
        
        self.experts = nn.ModuleList([
            MoEExpert(input_size, hidden_size) for _ in range(num_experts)
        ])
        
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.softmax = nn.Softmax(1)
        
        self.merger = None
        if with_smart_merger:
            self.merger = SmartMerger(output_size)

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1: return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def forward(self, x, shape_info=None):
        # x: (B, L, C) - Full Image Tokens
        # shape_info: (B, H, W, C)
        B, H, W, C = shape_info
        
        # 1. Partition into Windows for Gating
        x_img = x.view(B, H, W, C)
        
        windows, (wh, ww) = window_partition(x_img, self.window_size) # (B*nH*nW, Wh, Ww, C)
        # Flatten window pixels
        windows = windows.view(-1, self.window_size*self.window_size, C) # (N_windows, L_win, C)
        
        # 2. Per-Window Gating
        # Pool features: Mean over window pixels
        gate_inp = windows.mean(dim=1) # (N_windows, C)
        
        logits = gate_inp @ self.w_gate
        
        # Top-K
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)
        
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        
        # Loss
        load = (gates > 0).sum(0)
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        
        # 3. Dispatch
        dispatcher = PerWindowDispatcher(self.num_experts, gates, self.k)
        expert_inputs = dispatcher.dispatch(windows)
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        
        # 4. Combine & Merge
        y = dispatcher.combine(expert_outputs, shape_info, merger=self.merger)
        
        return y, loss