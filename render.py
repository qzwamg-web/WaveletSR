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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

import matplotlib.pyplot as plt

def visualize_depth(depth, min_depth=None, max_depth=None):
    """
    将单通道深度图转换为伪彩色图 (Turbo 或 Jet 风格)
    """
    depth = depth.cpu().numpy().squeeze()
    if min_depth is None:
        min_depth = depth.min()
    if max_depth is None:
        max_depth = depth.max()
        
    # 归一化到 0-1
    depth_norm = (depth - min_depth) / (max_depth - min_depth + 1e-5)
    depth_norm = depth_norm.clip(0, 1)
    
    # 使用 matplotlib 获取颜色映射
    cmap = plt.get_cmap('turbo') # 'turbo' 渲染效果非常好，颜色对比明显
    depth_color = cmap(depth_norm)[:, :, :3] # 取 RGB 通道
    
    # 转换回 torch Tensor [3, H, W]
    return torch.from_numpy(depth_color).permute(2, 0, 1)
def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    # [新增] 深度图路径
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    depth_vis_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth_vis")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(depth_vis_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # 调用渲染函数
        view.image_height = 912
        view.image_height =1368

        render_results = render(view, gaussians, pipeline, background)
        
        rendering = render_results["render"]
        # [提取深度] render_results["render_depth"] 形状通常为 [1, H, W]
        depth = render_results["render_depth"]
        
        # 1. 保存 RGB 图像
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
        # 2. 保存原始深度数据 (以 .pt 或 .npy 保存，保留绝对精度)
        # 如果你想存成单通道图，也可以直接保存，但需要自己处理归一化
        torch.save(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".pt"))
        
        # 3. 深度图可视化并保存
        depth_vis = visualize_depth(depth)
        torchvision.utils.save_image(depth_vis, os.path.join(depth_vis_path, '{0:05d}'.format(idx) + ".png"))

# def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
#     render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
#     gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

#     makedirs(render_path, exist_ok=True)
#     makedirs(gts_path, exist_ok=True)

#     for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
#         rendering = render(view, gaussians, pipeline, background)["render"]
#         gt = view.original_image[0:3, :, :]
#         torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
#         torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, optimizer_type="default")
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)