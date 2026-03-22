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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

C1 = 0.01 ** 2
C2 = 0.03 ** 2

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)



from typing import Literal, Optional
from torch import Tensor, nn

# class LogL1(nn.Module):
#     """Log-L1 loss"""

#     def __init__(
#         self, implementation: Literal["scalar", "per-pixel"] = "scalar", **kwargs
#     ):
#         super().__init__()
#         self.implementation = implementation

#     def forward(self, pred, gt):
#         if self.implementation == "scalar":
#             return torch.log(1 + torch.abs(pred - gt)).mean()
#         else:
#             return torch.log(1 + torch.abs(pred - gt))

# class EdgeAwareLogL1(nn.Module):
#     """Gradient aware Log-L1 loss"""

#     def __init__(
#         self, implementation: Literal["scalar", "per-pixel"] = "scalar", **kwargs
#     ):
#         super().__init__()
#         self.implementation = implementation
#         self.logl1 = LogL1(implementation="per-pixel")

#     def forward(self, pred: Tensor, gt: Tensor, rgb: Tensor, mask: Optional[Tensor]):
#         logl1 = self.logl1(pred, gt)

#         grad_img_x = torch.mean(
#             torch.abs(rgb[..., :, :-1, :] - rgb[..., :, 1:, :]), -1, keepdim=True
#         )
#         grad_img_y = torch.mean(
#             torch.abs(rgb[..., :-1, :, :] - rgb[..., 1:, :, :]), -1, keepdim=True
#         )
#         lambda_x = torch.exp(-grad_img_x)
#         lambda_y = torch.exp(-grad_img_y)

#         loss_x = lambda_x * logl1[..., :, :-1, :]
#         loss_y = lambda_y * logl1[..., :-1, :, :]

#         if self.implementation == "per-pixel":
#             if mask is not None:
#                 loss_x[~mask[..., :, :-1, :]] = 0
#                 loss_y[~mask[..., :-1, :, :]] = 0
#             return loss_x[..., :-1, :, :] + loss_y[..., :, :-1, :]

#         if mask is not None:
#             assert mask.shape[:2] == pred.shape[:2]
#             loss_x = loss_x[mask[..., :, :-1, :]]
#             loss_y = loss_y[mask[..., :-1, :, :]]

#         if self.implementation == "scalar":
#             return loss_x.mean() + loss_y.mean()



class LogL1(nn.Module):
    """Log-L1 loss"""
    def __init__(self, implementation: Literal["scalar", "per-pixel"] = "scalar"):
        super().__init__()
        self.implementation = implementation

    def forward(self, pred, gt):
        if self.implementation == "scalar":
            return torch.log(1 + torch.abs(pred - gt)).mean()
        else:
            return torch.log(1 + torch.abs(pred - gt))

class EdgeAwareLogL1(nn.Module):
    """边缘感知的 Log-L1 损失，适配 [C, H, W] 格式"""

    def __init__(self, implementation: Literal["scalar", "per-pixel"] = "scalar"):
        super().__init__()
        self.implementation = implementation
        self.logl1 = LogL1(implementation="per-pixel")

    def forward(self, pred: Tensor, gt: Tensor, rgb: Tensor, mask: Optional[Tensor] = None):
        # 1. 计算基础的 LogL1 差异图
        # 假设 pred, gt 形状为 [1, H, W]，logl1 形状也将是 [1, H, W]
        logl1 = self.logl1(pred, gt)

        # 2. 计算 RGB 图像的梯度作为权重 (边缘平滑约束)
        # rgb 形状为 [3, H, W]
        
        # 计算 X 方向（宽度方向）的颜色梯度：针对最后一个维度 (-1) 切片
        # 我们对通道维 (-3) 取平均，得到单通道的梯度权重
        grad_img_x = torch.mean(
            torch.abs(rgb[..., :, :, :-1] - rgb[..., :, :, 1:]), 
            dim=-3, keepdim=True
        )
        # 计算 Y 方向（高度方向）的颜色梯度：针对倒数第二个维度 (-2) 切片
        grad_img_y = torch.mean(
            torch.abs(rgb[..., :, :-1, :] - rgb[..., :, 1:, :]), 
            dim=-3, keepdim=True
        )
        
        # 边缘越强（梯度越大），lambda 越小，从而降低该处的损失权重，允许深度在此处有不连续性
        lambda_x = torch.exp(-grad_img_x)
        lambda_y = torch.exp(-grad_img_y)

        # 3. 将权重应用到深度损失上
        # 注意：logl1 也需要进行相应的切片以匹配 lambda 的尺寸
        loss_x = lambda_x * logl1[..., :, :, :-1]
        loss_y = lambda_y * logl1[..., :, :-1, :]

        # 4. 处理 Mask (如果有)
        if mask is not None:
            # 确保 mask 也是 [1, H, W] 并在对应维度切片
            if self.implementation == "per-pixel":
                loss_x[~mask[..., :, :, :-1]] = 0
                loss_y[~mask[..., :, :-1, :]] = 0
            else:
                loss_x = loss_x[mask[..., :, :, :-1]]
                loss_y = loss_y[mask[..., :, :-1, :]]

        # 5. 返回结果
        if self.implementation == "per-pixel":
            # 返回平滑项结果 (为了简化，这里只返回相加后的部分区域)
            return loss_x[..., :, :-1, :] + loss_y[..., :, :, :-1]

        if self.implementation == "scalar":
            # 通常返回两个方向梯度的平均值之和
            return loss_x.mean() + loss_y.mean()

