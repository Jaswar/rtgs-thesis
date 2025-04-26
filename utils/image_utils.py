#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2, mask):
    # make sure the input is batched
    if len(img1.shape) == 3:
        img1 = img1.unsqueeze(0)
    if len(img2.shape) == 3:
        img2 = img2.unsqueeze(0)
    if len(mask.shape) == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    img1 = img1.flatten(1)
    img2 = img2.flatten(1)
    mask = mask.flatten(1).repeat(1, 3)
    mask = torch.where(mask!=0,True,False)
    img1 = img1 * mask
    img2 = img2 * mask
    
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse.float()))
    return psnr
    # mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    # return 20 * torch.log10(1.0 / torch.sqrt(mse))

def easy_cmap(x: torch.Tensor):
    x_rgb = torch.zeros((3, x.shape[0], x.shape[1]), dtype=torch.float32, device=x.device)
    x_max, x_min = x.max(), x.min()
    x_normalize = (x - x_min) / (x_max - x_min)
    x_rgb[0] = torch.clamp(x_normalize, 0, 1)
    x_rgb[1] = torch.clamp(x_normalize, 0, 1)
    x_rgb[2] = torch.clamp(x_normalize, 0, 1)
    return x_rgb

