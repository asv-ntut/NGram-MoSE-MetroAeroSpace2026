import torch
import torch.nn as nn
import piq
import torchvision
import torchvision.models

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Use VGG19 pre-trained on ImageNet
        vgg = torchvision.models.vgg19(pretrained=True)
        # Extract features up to ReLU 5_4 (index 35)
        self.features = nn.Sequential(*list(vgg.features.children())[:36]).eval()
        
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False
            
        self.l1 = nn.L1Loss()
        
        # Normalization for VGG (ImageNet mean/std)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr, hr):
        # Assume input is 0-1. Normalize to VGG expected input
        sr = (sr - self.mean) / self.std
        hr = (hr - self.mean) / self.std
        
        sr_feat = self.features(sr)
        hr_feat = self.features(hr)
        
        return self.l1(sr_feat, hr_feat)

class FusionLoss(nn.Module):
    def __init__(self, w_pixel=1.0, w_ssim=1.0, w_ncc=1.0, w_moe=0.2, w_perceptual=0.1):
        super().__init__()
        self.w_pixel = w_pixel
        self.w_ssim = w_ssim
        self.w_ncc = w_ncc
        self.w_moe = w_moe
        self.w_perceptual = w_perceptual
        
        self.l1 = nn.L1Loss()
        self.ssim = piq.SSIMLoss(data_range=1.0)
        
        if self.w_perceptual > 0:
            self.vgg = VGGLoss()
        
    def ncc_loss(self, x, y):
        # Swin2-MoSE NCC Formula
        mean_x = torch.mean(x, dim=[1,2,3], keepdim=True)
        mean_y = torch.mean(y, dim=[1,2,3], keepdim=True)
        
        vx = x - mean_x
        vy = y - mean_y
        
        num = torch.sum(vx * vy, dim=[1,2,3])
        den = torch.sqrt(torch.sum(vx**2, dim=[1,2,3]) * torch.sum(vy**2, dim=[1,2,3]) + 1e-8)
        
        cc = num / den
        return 1 - torch.mean(cc)

    def forward(self, sr, hr, moe_loss=0):
        loss_dict = {}
        total_loss = 0
        
        if self.w_pixel > 0:
            l_pix = self.l1(sr, hr)
            total_loss += self.w_pixel * l_pix
            loss_dict['pixel'] = l_pix.item()
            
        if self.w_ssim > 0:
            l_ssim = self.ssim(sr.clamp(0, 1), hr.clamp(0, 1))
            total_loss += self.w_ssim * l_ssim
            loss_dict['ssim'] = l_ssim.item()
            
        if self.w_ncc > 0:
            l_ncc = self.ncc_loss(sr, hr)
            total_loss += self.w_ncc * l_ncc
            loss_dict['ncc'] = l_ncc.item()
            
        if self.w_perceptual > 0:
            l_percep = self.vgg(sr, hr)
            total_loss += self.w_perceptual * l_percep
            loss_dict['perceptual'] = l_percep.item()
            
        if self.w_moe > 0:
            if torch.is_tensor(moe_loss):
                 total_loss += self.w_moe * moe_loss
                 loss_dict['moe'] = moe_loss.item()
            elif moe_loss > 0: # float
                 total_loss += self.w_moe * moe_loss
                 loss_dict['moe'] = moe_loss
            
        return total_loss, loss_dict