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

import os, math
import numpy as np
import torch
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.read_write_model import Camera as ColmapCamera, Image as ColmapImage, write_cameras_binary, write_images_binary, rotmat2qvec
from scene import Scene
import copy

# NOTE: You MUST have scipy installed for SLERP rotation interpolation.
from scipy.spatial.transform import Rotation as R, Slerp

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def get_camera_center(view):
    """Helper to extract camera center from w2c matrix."""
    w2c = view.world_view_transform.detach().cpu().numpy().T
    R_wc = w2c[:3, :3]
    t_wc = w2c[:3, 3]
    # C = -R^T * t
    center = -np.dot(R_wc.T, t_wc)
    return center, R_wc


def interpolate_camera(cam1, cam2, alpha):
    """
    Interpolates between two cameras.
    alpha = 0.0 returns cam1, alpha = 1.0 returns cam2.
    """
    c1_center, c1_R = get_camera_center(cam1)
    c2_center, c2_R = get_camera_center(cam2)

    # 1. Interpolate Translation (Camera Center)
    new_center = (1.0 - alpha) * c1_center + alpha * c2_center

    # 2. Interpolate Rotation (SLERP)
    rots = R.from_matrix([c1_R, c2_R])
    slerp = Slerp([0.0, 1.0], rots)
    new_R = slerp(alpha).as_matrix()

    # 3. Reconstruct world_view_transform
    new_t = -np.dot(new_R, new_center)
    new_w2c = np.eye(4)
    new_w2c[:3, :3] = new_R
    new_w2c[:3, 3] = new_t

    # 4. Create new camera object (deep copy to keep intrinsics and structure)
    new_cam = copy.deepcopy(cam1)
    
    # Update transformation matrices
    new_cam.world_view_transform = torch.tensor(new_w2c.T, dtype=torch.float32, device=cam1.world_view_transform.device)
    new_cam.full_proj_transform = (new_cam.world_view_transform.unsqueeze(0).bmm(new_cam.projection_matrix.unsqueeze(0))).squeeze(0)
    new_cam.camera_center = new_cam.world_view_transform.inverse()[3, :3]
    
    # Interpolate FoV/Intrinsics if needed
    new_cam.FoVx = (1.0 - alpha) * cam1.FoVx + alpha * cam2.FoVx
    new_cam.FoVy = (1.0 - alpha) * cam1.FoVy + alpha * cam2.FoVy

    return new_cam


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    # Check if interpolation is requested via environment variable
    inter_num = int(os.environ.get("GT_CAMERA_INTER_NUM", 0))
    if inter_num > 0:
        print(f"--- Interpolation Active: Inserting {inter_num} camera(s) between existing views ---")

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    export_test_images_bin = (name == "test")
    
    if export_test_images_bin:
        colmap_cameras_bin_path = os.path.join(model_path, name, "ours_{}".format(iteration), "cameras.bin")
        colmap_images_bin_path = os.path.join(model_path, name, "ours_{}".format(iteration), "images.bin")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    colmap_images = {} if export_test_images_bin else None
    colmap_cameras = {} if export_test_images_bin else None

    # Construct the final list of cameras to render
    render_cameras = []
    
    for idx in range(len(views)):
        # Always add the original camera
        base_name = os.path.splitext(str(views[idx].image_name))[0]
        views[idx].image_name = base_name # Ensure no extension here for clean handling later
        render_cameras.append(views[idx])
        
        # If not the last image, insert interpolated cameras
        if inter_num > 0 and idx < len(views) - 1:
            cam1 = views[idx]
            cam2 = views[idx + 1]
            
            for step in range(1, inter_num + 1):
                alpha = step / float(inter_num + 1)
                inter_cam = interpolate_camera(cam1, cam2, alpha)
                # Give it a unique name
                inter_cam.image_name = f"{base_name}_inter_{step}"
                # It doesn't have a real GT image, so we set it to None or zeros
                inter_cam.original_image = torch.zeros_like(cam1.original_image) 
                render_cameras.append(inter_cam)


    # Now render the constructed list
    for idx, view in enumerate(tqdm(render_cameras, desc="Rendering progress")):
        
        out_filename = str(view.image_name) + ".JPG"

        if export_test_images_bin:
            w2c = view.world_view_transform.detach().cpu().numpy().T
            R_wc = w2c[:3, :3]
            t_wc = w2c[:3, 3]
            qvec = rotmat2qvec(R_wc)
            image_id = int(idx + 1)
            camera_id = image_id
            width = int(view.image_width)
            height = int(view.image_height)
            fx = width / (2.0 * np.tan(float(view.FoVx) * 0.5))
            fy = height / (2.0 * np.tan(float(view.FoVy) * 0.5))
            cx = width / 2.0
            cy = height / 2.0
            
            colmap_cameras[camera_id] = ColmapCamera(
                id=camera_id,
                model="PINHOLE",
                width=width,
                height=height,
                params=np.asarray([fx, fy, cx, cy], dtype=np.float64),
            )
            
            colmap_images[image_id] = ColmapImage(
                id=image_id,
                qvec=np.asarray(qvec, dtype=np.float64),
                tvec=np.asarray(t_wc, dtype=np.float64),
                camera_id=camera_id,
                name=out_filename, 
                xys=np.empty((0, 2), dtype=np.float64),
                point3D_ids=np.empty((0,), dtype=np.int64),
            )

        rendering = render(view, gaussians, pipeline, background)["render"] 
        
        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, out_filename))
        
        # Only save GT if it's an original image (not interpolated)
        if "_inter_" not in view.image_name:
             gt = view.original_image[0:3, :, :]
             if args.train_test_exp:
                 gt = gt[..., gt.shape[-1] // 2:]
             torchvision.utils.save_image(gt, os.path.join(gts_path, out_filename))


    if export_test_images_bin:
        write_cameras_binary(colmap_cameras, colmap_cameras_bin_path)
        write_images_binary(colmap_images, colmap_images_bin_path)
        print(f"Saved COLMAP cameras.bin to {colmap_cameras_bin_path}")
        print(f"Saved COLMAP images.bin to {colmap_images_bin_path}")


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)


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

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)


    # CUDA_VISIBLE_DEVICES=1 GT_CAMERA_INTER_NUM=1 python render.py -m /gdata/cold1/fujiaye/qzwang/test/track2/EastResearchAreas