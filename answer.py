# [新增/引用] 确保导入了读取 COLMAP 的基础函数
from scene.dataset_readers import read_extrinsics_binary, read_intrinsics_binary, readColmapCameras
from utils.camera_utils import loadCam
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
import numpy as np       # [新增]
import skimage.io        # [新增]
import warnings          # [新增]


def render_with_external_cameras(model_path, external_cam_path, gaussians, pipeline, background):
    # 1. 定义输出目录
    scene_name = os.path.basename(os.path.normpath(model_path))
    submission_path = os.path.join("/gdata/cold1/fujiaye/qzwang/ntire/outputs", "submission", scene_name)
    makedirs(submission_path, exist_ok=True)

    # 2. 读取外部相机参数 (参考你的截图逻辑)
    # 假设 external_cam_path 指向包含 sparse/0 的目录
    sparse_dir = os.path.join(external_cam_path, "sparse", "0")
    cam_extrinsics = read_extrinsics_binary(os.path.join(sparse_dir, "images.bin"))
    cam_intrinsics = read_intrinsics_binary(os.path.join(sparse_dir, "cameras.bin"))
    image_path = os.path.join(external_cam_path, "images")
    
    # 获取 cam_infos 列表
    cam_infos = readColmapCameras(cam_extrinsics=cam_extrinsics, 
                                  cam_intrinsics=cam_intrinsics, 
                                  images_folder=image_path)
    # 按名称排序以保证索引稳定性
    # cam_infos = sorted(cam_infos, key=lambda x: x.image_name)

    # 3. 指定渲染索引
    target_indices = [6, 16, 22, 25, 39, 50, 51, 52, 66, 71]

    for idx in target_indices:
        if idx >= len(cam_infos): continue
        
        # 将读取的 cam_info 转换为渲染所需的 camera 对象
        # loadCam 是 3DGS 库中将 cam_info 转换为渲染用 Camera 对象的函数
        view = loadCam(args, idx, cam_infos[idx], 1.0)
        view.image_height = 912
        view.image_width = 1368
        # 渲染
        render_results = render(view, gaussians, pipeline, background)
        
        # --- 保存逻辑 (保持你原有的) ---
        img_basename = os.path.splitext(cam_infos[idx].image_name)[0]
        
        # RGB 保存
        rendered_img = render_results["render"].clamp(0, 1).permute(1, 2, 0).data.cpu().numpy() * 255.0
        skimage.io.imsave(os.path.join(submission_path, f"{img_basename}.JPG"), rendered_img.astype('uint8'))
        
        # 深度保存
        # depth_np = render_results["render_depth"][0, ...].data.cpu().numpy() * 256.0
        # skimage.io.imsave(os.path.join(submission_path, f"{img_basename}_depth.png"), depth_np.astype('uint16'))

        depth_np = render_results["render_depth"][0, ...].data.cpu().numpy()
        depth_scaled = (depth_np) * 256.0 * 256.0
        depth_final = np.clip(depth_scaled, 0, 65535).astype('uint16')
        skimage.io.imsave(os.path.join(submission_path, f"{img_basename}_depth.png"), depth_final)



# 在主函数中调用
if __name__ == "__main__":
    # ... 原有参数解析 ...
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    parser.add_argument("--ext_cam_path", type=str, required=True) # [新增参数]
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    # 初始化 Gaussians
    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(os.path.join(args.model_path, "point_cloud", "iteration_30000", "point_cloud.ply"))
    
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    # 渲染
    render_with_external_cameras(args.model_path, args.ext_cam_path, gaussians, pipeline.extract(args), background)


# python answer.py -m /gdata/cold1/fujiaye/qzwang/ntire/outputs/NorthAreas_track1 --ext_cam_path /gdata/cold1/fujiaye/qzwang/test/track1/NorthAreas 
# 
# NorthAreas EastResearchAreas WestAccommodationAreas  WestResearchAreas  WestTeachingAreas  