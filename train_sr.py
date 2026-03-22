# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os, glob, math
import numpy as np
import torch
import random
from random import randint
from utils.loss_utils import l1_loss, EdgeAwareLogL1
from fused_ssim import fused_ssim as fast_ssim
from gaussian_renderer import render
from torch import autocast
import sys
import copy
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
# from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
# from scipy.spatial.transform import Rotation as R, Slerp
import torchvision
from scene.cameras import Camera
from PIL import Image
from utils.general_utils import PILtoTorch
try:
    # from torch.utils.tensorboard import SummaryWriter
    from tensorboardX import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    
##### Stable SR usage #####
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from utils.stable_sr_utils import instantiate_from_config
from utils.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization
from contextlib import nullcontext
from tqdm import tqdm, trange
from einops import rearrange, repeat
from utils.util_image import ImageSpliterTh
import torch.nn.functional as F
from pathlib import Path
from PIL import ImageFilter
import torchvision.transforms as transforms

def compute_haar_ll_band(img_tensor):
    c = img_tensor.shape[0]
    weight = torch.tensor([[[[0.25, 0.25], 
                             [0.25, 0.25]]]], device=img_tensor.device, dtype=img_tensor.dtype)
    weight = weight.repeat(c, 1, 1, 1)
    img_unsqueeze = img_tensor.unsqueeze(0)
    ll_band = F.conv2d(img_unsqueeze, weight, stride=2, groups=c).squeeze(0)
    return ll_band
  
def get_edges(image):
    image_pil = transforms.ToPILImage()(image)
    image_gray = image_pil.convert('L')
    image_edges = image_gray.filter(ImageFilter.FIND_EDGES)
    image_edges_tensor = transforms.ToTensor()(image_edges)
    return image_edges_tensor

def normalize(value_tensor):
    value_tensor[value_tensor.isnan()] = 0
    valid_indices = (value_tensor > 0)
    valid_value = value_tensor[valid_indices].to(torch.float32)
    ret_value = torch.zeros_like(value_tensor, dtype=torch.float32)
    ret_value[valid_indices] = valid_value / torch.mean(valid_value)

    return ret_value


def compute_edge_score(camlist, edge_losses, gaussians, pipe, bg):
    num_points = len(gaussians.get_xyz)
    gaussian_importance = torch.zeros(num_points, device="cuda", dtype=torch.float32)
    visibility_filter_all = torch.zeros(num_points, device="cuda", dtype=bool)
    for view in range(len(camlist)):
        my_viewpoint_cam = camlist[view]
        pixel_weights = edge_losses[view].cuda()
        render_pkg = render(my_viewpoint_cam, gaussians, pipe, bg, pixel_weights=pixel_weights)

        loss_accum = normalize(render_pkg["accum_weights"])
        visibility_filter = render_pkg["visibility_filter"].detach()

        gaussian_importance[visibility_filter] += loss_accum[visibility_filter] / len(camlist)
        visibility_filter_all[visibility_filter] = True

    gaussian_importance[visibility_filter_all] = gaussian_importance[visibility_filter_all]
    return gaussian_importance

def prepare_training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    
    scene = Scene(dataset, gaussians, load_iteration=30000, shuffle=False)
    scene.model_path = args.output_folder
    dataset_name = os.path.basename(dataset.source_path)
    dataset.model_path = args.output_folder #os.path.join(args.output_folder, dataset_name)
        
    tb_writer = prepare_output_and_logger(dataset)
    scene.model_path = dataset.model_path
    
    gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), dtype=torch.float32, device="cuda")
    gaussians.training_setup(opt)
    print("--- after loading pretrain points:", gaussians._xyz.shape[0])

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)    

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    
    out_dict = {"scene": scene, "gaussians": gaussians, "tb_writer": tb_writer}
    return out_dict

def training_with_iters(in_dict, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args, SR_iter=0):
    scene = in_dict['scene']
    gaussians = in_dict['gaussians']
    tb_writer = in_dict['tb_writer']
    
    # 将总迭代次数平分到所有的 args.ddpm_steps 中
    chunk_size = opt.iterations // args.ddpm_steps
    step_idx = args.ddpm_steps - SR_iter - 1
    start_iter = step_idx * chunk_size
    end_iter = start_iter + chunk_size
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    gt_lr_cache = {}

    trainCameras = scene.getTrainCameras().copy()
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    all_edges = []
    for view in trainCameras: 
        edges_loss = get_edges(view.original_image).squeeze().cuda()
        edges_loss_norm = (edges_loss - torch.min(edges_loss)) / (torch.max(edges_loss) - torch.min(edges_loss))
        all_edges.append(edges_loss_norm.cpu())
        
    my_viewpoint_stack = scene.getTrainCameras().copy()
    edges_stack = all_edges.copy()

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(start_iter, end_iter), desc=f"Training progress (Step {3-SR_iter})")
    
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    # 循环区间使用绝对数值，保证高斯的全局调度正常
    for iteration in range(start_iter + 1, end_iter + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = trainCameras.copy() 
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        _ = viewpoint_indices.pop(rand_idx)
            
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss_hr = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss = loss_hr
        
        if args.fidelity_train_en:
            cam_name = viewpoint_cam.image_name
            if cam_name not in gt_lr_cache:
                gt_path = os.path.join(dataset.source_path, 'images', cam_name + '.JPG')
                image_gt_lr_pil = Image.open(gt_path)
                w_lr, h_lr = image_gt_lr_pil.size
                gt_lr_tensor = PILtoTorch(image_gt_lr_pil, (w_lr, h_lr)).cuda()
                gt_lr_cache[cam_name] = gt_lr_tensor
            image_gt_lr = gt_lr_cache[cam_name]
            
            image_lr = torch.nn.functional.interpolate(
                image.unsqueeze(0), 
                size=(image_gt_lr.shape[1], image_gt_lr.shape[2]), 
                mode='bicubic', 
                antialias=True
            ).squeeze(0)
            ll_render = compute_haar_ll_band(image_lr)
            ll_gt = compute_haar_ll_band(image_gt_lr)
            loss_ll_l1 = l1_loss(ll_render, ll_gt)
            ssim_value = fast_ssim(ll_render.unsqueeze(0), ll_gt.unsqueeze(0))
            loss_lr = (1.0 - opt.lambda_dssim) * loss_ll_l1 + opt.lambda_dssim * (1.0 - ssim_value)
            loss += loss_lr * args.wt_lr
                    
        loss.backward()
        iter_end.record()
        
        # 3. 在当前 chunk 的最后一次迭代进行全视角保存
        if iteration == end_iter:
            training_views_folder = os.path.join(args.output_folder, 'training_views')
            os.makedirs(training_views_folder, exist_ok=True)
            
            step_val = 3 - SR_iter
            step_dir = os.path.join(args.outdir, f'step_{step_val}')
            os.makedirs(step_dir, exist_ok=True)
                
            for i in range(len(trainCameras)):                
                cam = trainCameras[i]
                renderpkg = render(cam, gaussians, pipe, bg)
                rendering = renderpkg["render"]
                
                view_name = os.path.join(training_views_folder, cam.image_name + ".png")
                torchvision.utils.save_image(rendering, view_name)

                file_name = os.path.join(step_dir, cam.image_name + f"_step_{step_val}.png")
                torchvision.utils.save_image(rendering, file_name)


        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == end_iter:
                progress_bar.close()

            # Log and save 
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), dataset, gaussians.get_xyz.shape[0])
            if (iteration in saving_iterations) or (iteration == end_iter):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if opt.densify_from_iter < iteration < opt.densify_until_iter:
                gaussians.add_densification_stats_abs(viewspace_point_tensor, visibility_filter)

                if iteration % opt.densification_interval == 0:
                    num_cams = args.cams
                    if args.cams == -1 or (iteration % 3000 == 400 and iteration < 9000):
                        num_cams = len(trainCameras.copy())
                    edge_losses = []
                    camlist = []
                    for _ in range(num_cams):
                        if not my_viewpoint_stack:
                            my_viewpoint_stack = trainCameras.copy()
                            edges_stack = all_edges.copy()
                        camlist.append(my_viewpoint_stack.pop())
                        edge_losses.append(edges_stack.pop())
                    gaussian_importance = compute_edge_score(camlist, edge_losses, gaussians, pipe, bg)

                    startI = opt.densify_from_iter
                    endI = opt.densify_until_iter - 500
                    rate = (iteration - startI) / (endI - startI)
                    if rate >= 1:
                        budget = int(opt.budget)
                    else:
                        budget = int(math.sqrt(rate) * opt.budget)

                    gaussians.densify_and_prune_Improved(gaussian_importance, 0.005, budget, opt, iteration)
                
                if iteration % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity(0.05)

                if iteration % opt.opacity_reset_interval == 300 and iteration < 9000:
                    gaussians.only_prune(0.2, True)

            # 4. 优化器正常执行
            if iteration <= opt.iterations:
                if opt.optimizer_type == "default":
                    if iteration <= 20000:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none=True)
                        gaussians.shoptimizer.step()
                        gaussians.shoptimizer.zero_grad(set_to_none=True)
                    elif iteration <= 24000:
                        if iteration % 5 == 0:
                            gaussians.optimizer.step()
                            gaussians.optimizer.zero_grad(set_to_none=True)
                            gaussians.shoptimizer.step()
                            gaussians.shoptimizer.zero_grad(set_to_none=True)
                    else:
                        if iteration % 20 == 0:
                            gaussians.optimizer.step()
                            gaussians.optimizer.zero_grad(set_to_none=True)
                            gaussians.shoptimizer.step()
                            gaussians.shoptimizer.zero_grad(set_to_none=True)
                elif opt.optimizer_type == "sparse_adam":
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    
    out_dict = {"scene": scene, "gaussians": gaussians, "tb_writer": tb_writer}
    
    return out_dict

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def prepare_model(opt):
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.configs = config
    
    vqgan_config = OmegaConf.load("configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
    vq_model = load_model_from_config(vqgan_config, opt.vqgan_ckpt)
    vq_model = vq_model.to(device)
    vq_model.decoder.fusion_w = opt.dec_w
    
    model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3)
       
    out_dict = {'model': model, 'vq_model': vq_model}
    return out_dict

def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]   #[250,]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)

def read_image(im_path):
    im = np.array(Image.open(im_path).convert("RGB"))
    im = im.astype(np.float32)/255.0
    im = im[None].transpose(0,3,1,2)
    im = (torch.from_numpy(im) - 0.5) / 0.5
    return im.cuda()

def visualize_image(latent, rgb_patch, model_dict, out_img_name=None):
    vq_model = model_dict['vq_model']
    model = model_dict['model']
    _, enc_fea_lq = vq_model.encode(rgb_patch)
    x_samples = vq_model.decode(latent * 1. / model.scale_factor, enc_fea_lq)
    x_samples = wavelet_reconstruction(x_samples, rgb_patch)
    im_sr = torch.clamp((x_samples+1.0)/2.0, min=0.0, max=1.0)
    out = Image.fromarray(np.uint8(im_sr[0, ].permute(1,2,0).cpu().numpy()*255))
    
    if out_img_name is not None:        
        out.save(out_img_name)
    return out
    
def train_proposed(dataset, op, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):    
    ####################################
    # Set up for Stable SR
    ####################################
    print('>>>>>>>>>>color correction>>>>>>>>>>>')
    if args.colorfix_type == 'adain':
        print('Use adain color correction')
    elif args.colorfix_type == 'wavelet':
        print('Use wavelet color correction')
    else:
        print('No color correction')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    
    #############################################
    # load StableSR model and scheduler
    #############################################
    # Check input images
    os.makedirs(args.outdir, exist_ok=True)
    outpath = args.outdir
    batch_size = args.n_samples
    images_path_ori = sorted(glob.glob(os.path.join(args.init_img, "*")))
    images_path = np.array(copy.deepcopy(images_path_ori))
    print(args.init_img, args.source_path)
    sr_indices = np.arange(len(images_path))
    images_path = images_path[sr_indices[:]]
    print(f"Found {len(images_path)} inputs.")
    
    # Prepare model
    out_dict = prepare_model(args)
    model = out_dict['model']
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    sqrt_alphas_cumprod = copy.deepcopy(model.sqrt_alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = copy.deepcopy(model.sqrt_one_minus_alphas_cumprod)
    
    # Modify scheduler for fewer steps
    use_timesteps = set(space_timesteps(1000, [args.ddpm_steps]))
    last_alpha_cumprod = 1.0
    new_betas = []
    timestep_map = []
    for i, alpha_cumprod in enumerate(model.alphas_cumprod):
        if i in use_timesteps:
            new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
            last_alpha_cumprod = alpha_cumprod
            timestep_map.append(i)
    new_betas = [beta.data.cpu().numpy() for beta in new_betas]
    model.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas))
    model.num_timesteps = 1000
    model.ori_timesteps = list(use_timesteps)
    model.ori_timesteps.sort()
    model = model.to(device)
    
    # Add model and args to out_dict
    out_dict['model'] = model
    out_dict['args'] = args
    precision_scope = autocast if args.precision == "autocast" else nullcontext
    
    #############################################
    # Loading scene and Gaussians
    #############################################    
    op.densify_until_iter = args.densify_end
    input_dict = prepare_training(dataset, op, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args)
    scene = input_dict["scene"]
    trainCameras = scene.getTrainCameras()
        
    #############################################
    # Prepare for SR method
    #############################################
    with model.ema_scope():     
        seed_everything(args.seed)
        
        imgs_per_batch = batch_size
        loop_img_time = len(images_path) // imgs_per_batch
        one_more_time = (len(images_path) % imgs_per_batch) > 0        
        loop_img_time += int(one_more_time)
      
        #############################################
        # Loop by denoising steps
        #############################################
        for iteration in range(args.ddpm_steps-1, -1, -1):
            print("************** Diffusion step ", 3-iteration, "**************")
            for loop_id in range(loop_img_time):
                if loop_id == loop_img_time - 1:
                    images_path_small = images_path[loop_id*imgs_per_batch:]
                else:
                    images_path_small = images_path[loop_id*imgs_per_batch : (loop_id+1)*imgs_per_batch]
                
                im_lq_bs = []
                im_path_bs = []
                for img_id in range(len(images_path_small)):
                    cur_image = read_image(images_path_small[img_id])
                    size_min = min(cur_image.size(-1), cur_image.size(-2))
                    upsample_scale = max(args.input_size/size_min, args.upscale)
                    cur_image = F.interpolate(
                                cur_image,
                                size=(int(cur_image.size(-2)*upsample_scale),
                                        int(cur_image.size(-1)*upsample_scale)),
                                mode='bicubic',
                                )
                    cur_image = cur_image.clamp(-1, 1)                    
                    im_lq_bs.append(cur_image) 
                    im_path_bs.append(images_path_small[img_id]) 
                im_lq_bs = torch.cat(im_lq_bs, dim=0)                
                ori_h, ori_w = im_lq_bs.shape[2:]
                ref_patch=None
                if not (ori_h % 32 == 0 and ori_w % 32 == 0):
                    flag_pad = True
                    pad_h = ((ori_h // 32) + 1) * 32 - ori_h
                    pad_w = ((ori_w // 32) + 1) * 32 - ori_w
                    im_lq_bs = F.pad(im_lq_bs, pad=(0, pad_w, 0, pad_h), mode='reflect')
                else:
                    flag_pad = False
                    
                if iteration != args.ddpm_steps - 1:
                    #####################################################
                    # Load upsampled image, and encode to latent space
                    #####################################################
                    imgs = []
                    for img_id in range(len(im_path_bs)):
                        img_name = str(Path(im_path_bs[img_id]).name)
                        basename = os.path.splitext(os.path.basename(img_name))[0]
                        cur_id = loop_id * imgs_per_batch + img_id
                        prev_step_val = 3 - int(iteration) - 1
                        prev_step_dir = os.path.join(args.outdir, f'step_{prev_step_val}')
                        imgpath = os.path.join(prev_step_dir, trainCameras[cur_id].image_name + f"_step_{prev_step_val}.png")                      
                        cur_image = read_image(imgpath)
                        
                        # Add padding to loaded image
                        if not (ori_h % 32 == 0 and ori_w % 32 == 0):
                            pad_h = ((ori_h // 32) + 1) * 32 - ori_h
                            pad_w = ((ori_w // 32) + 1) * 32 - ori_w
                            cur_image = F.pad(cur_image, pad=(0, pad_w, 0, pad_h), mode='reflect')
                        imgs.append(cur_image)
                    imgs = torch.cat(imgs, dim=0)
                    
                with torch.no_grad():
                    with precision_scope("cuda"):
                        #############################################
                        # Start of loop for denoised images
                        #############################################
                        for img_id in range(len(im_path_bs)):
                            #############################################
                            # Split image to patches
                            #############################################
                            if im_lq_bs.shape[2] > args.vqgantile_size or im_lq_bs.shape[3] > args.vqgantile_size:
                                im_spliter = ImageSpliterTh(im_lq_bs[img_id].unsqueeze(0), args.vqgantile_size, args.vqgantile_stride, sf=1)
                                if iteration != args.ddpm_steps-1:
                                    im_spliter_x_tilda = ImageSpliterTh(imgs[img_id].unsqueeze(0), args.vqgantile_size, args.vqgantile_stride, sf=1)
                                #############################################
                                # Loop to process each patch in an image   
                                #############################################                        
                                for im_lq_pch, index_infos in im_spliter:
                                    if iteration == args.ddpm_steps-1:
                                        init_latent = model.get_first_stage_encoding(model.encode_first_stage(im_lq_pch))  
                                        text_init = ['']*args.n_samples
                                        semantic_c = model.cond_stage_model(text_init)
                                        noise = torch.randn_like(init_latent)
                                        t = repeat(torch.tensor([999]), '1 -> b', b=im_lq_pch.size(0))
                                        t = t.to(device).long()
                                        x_T = model.q_sample_respace(x_start=init_latent, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, 
                                                sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)
                                        _, x0_head = model.sample_canvas_one_iter(iteration=iteration, cond=semantic_c, struct_cond=init_latent, 
                                                                                    batch_size=im_lq_pch.size(0), timesteps=args.ddpm_steps, time_replace=args.ddpm_steps, 
                                                                                    x_T=x_T, tile_size=int(args.input_size/8), tile_overlap=args.tile_overlap, 
                                                                                    batch_size_sample=args.n_samples, return_x0=True)
                                    else:
                                        im_lq_pch_tilda, index_infos_tilda = next(im_spliter_x_tilda)
                                        x0_tilda_latent = model.get_first_stage_encoding(model.encode_first_stage(im_lq_pch_tilda))  
                                        text_init = ['']*args.n_samples
                                        semantic_c = model.cond_stage_model(text_init)
                                        init_latent = model.get_first_stage_encoding(model.encode_first_stage(im_lq_pch))  
                                        x_T_1 = model.sample_canvas_one_iter(iteration=iteration+1, cond=semantic_c, struct_cond=init_latent, 
                                                        batch_size=im_lq_pch.size(0), timesteps=args.ddpm_steps, time_replace=args.ddpm_steps, 
                                                        x_T=x_T, tile_size=int(args.input_size/8), tile_overlap=args.tile_overlap, 
                                                        batch_size_sample=args.n_samples, return_x0=False, x0_input=x0_tilda_latent)
                                        _, x0_head = model.sample_canvas_one_iter(iteration=iteration, cond=semantic_c, struct_cond=init_latent, 
                                                                                    batch_size=im_lq_pch.size(0), timesteps=args.ddpm_steps, time_replace=args.ddpm_steps, 
                                                                                    x_T=x_T_1, tile_size=int(args.input_size/8), tile_overlap=args.tile_overlap, 
                                                                                    batch_size_sample=args.n_samples, return_x0=True)
                                    # Decode the latent space to image space
                                    vq_model = out_dict['vq_model']
                                    _, enc_fea_lq = vq_model.encode(im_lq_pch)
                                    x_samples = vq_model.decode(x0_head * 1. / model.scale_factor, enc_fea_lq)
                                    
                                    if args.colorfix_type == 'adain':
                                        x_samples = adaptive_instance_normalization(x_samples, im_lq_pch)
                                    elif args.colorfix_type == 'wavelet':
                                        x_samples = wavelet_reconstruction(x_samples, im_lq_pch)
                                    im_spliter.update_gaussian(x_samples, index_infos)

                                im_sr = im_spliter.gather()
                                im_sr = torch.clamp((im_sr+1.0)/2.0, min=0.0, max=1.0)
                                
                                if upsample_scale > args.upscale:
                                    im_sr = F.interpolate(
                                                im_sr,
                                                size=(int(im_lq_bs.size(-2)*args.upscale/upsample_scale),
                                                    int(im_lq_bs.size(-1)*args.upscale/upsample_scale)),
                                                mode='bicubic',)
                                im_sr = torch.clamp(im_sr, min=0.0, max=1.0)
                                
                                if flag_pad:
                                    im_sr = im_sr[:, :, :ori_h, :ori_w, ]

                                im_sr = im_sr.cpu().numpy().transpose(0,2,3,1)*255   # b x h x w x c                                
                                img_name = str(Path(im_path_bs[img_id]).name)
                                basename = os.path.splitext(os.path.basename(img_name))[0]
                                step_val = 3 - int(iteration)
                                step_dir = os.path.join(args.outdir, f'step_{step_val}')
                                os.makedirs(step_dir, exist_ok=True)
                                outpath = os.path.join(step_dir, basename + f'_step_{step_val}.png')
                                Image.fromarray(im_sr[0, ].astype(np.uint8)).save(outpath)
                                print('Finished:', outpath)
                                
                                if iteration == 0:
                                    final_sr_path = os.path.join(args.outdir, 'final_sr_results')
                                    os.makedirs(final_sr_path, exist_ok=True)
                                    outpath = final_sr_path + '/' + basename + f'.png'
                                    Image.fromarray(im_sr[0, ].astype(np.uint8)).save(outpath)                        
                    #############################################
                    # End of loop for denoised images
                    #############################################                
            
            step_val = 3 - int(iteration)
            step_suffix = f"step_{step_val}"
            step_dir = os.path.join(args.outdir, step_suffix)

            print(f"--- 正在直接更新 Camera GT 图像 ({step_suffix}) ---")
            for cam in trainCameras:
                rgb_path = os.path.join(step_dir, f"{cam.image_name}_{step_suffix}.png")
                
                # Load RGB and convert to Torch
                img_transfer = Image.open(rgb_path).convert("RGB")
                width, height = img_transfer.size
                loaded_rgb = PILtoTorch(img_transfer, (width, height)).cuda()
                
                cam.update_gt_image(loaded_rgb.clone()) 
                
            # 清理显存碎片
            torch.cuda.empty_cache()

            #############################################
            # Train GS
            #############################################
            input_dict = training_with_iters(input_dict, dataset, op, pipe, testing_iterations, saving_iterations,
                                            checkpoint_iterations, checkpoint, debug_from, args, SR_iter=iteration)

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, dataset, num):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test= 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])    
                if config['name'] == 'train':
                    print("[ITER {}] Evaluating train: L1 {:.7f} PSNR {:.5f} N_GS {}".format(iteration, l1_test, psnr_test, num))
                    with open(os.path.join(dataset.model_path, 'log.txt'), 'a') as f:
                        f.write("[ITER {}] Evaluating train: L1 {:.7f} PSNR {:.5f} N_GS {}\n\n".format(iteration, l1_test, psnr_test, num))
                else:
                    print("\n[ITER {}] Evaluating test : L1 {:.7f} PSNR {:.5f} N_GS {}".format(iteration, l1_test, psnr_test, num))
                    with open(os.path.join(dataset.model_path, 'log.txt'), 'a') as f:
                        f.write("[ITER {}] Evaluating test : L1 {:.7f} PSNR {:.5f} N_GS {}\n".format(iteration, l1_test, psnr_test, num))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


def parse_args():
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--load_pretrain", action="store_true")
    parser.add_argument("--freeze_point", action="store_true")
    parser.add_argument("--SR_GS", action="store_true")
    parser.add_argument("--fidelity_train_en", action="store_true")
    parser.add_argument("--prune_init_en", action="store_true")
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--edge_aware_loss_en", action="store_true")
    parser.add_argument("--lpips_wt", type=float, default=0.2)
    parser.add_argument("--wt_lr", type=float, default=0.2)
    parser.add_argument("--densify_end", type=int, default=15000)

    parser.add_argument("--cams", type=int, default=10)

    #############################################
    #### From Stable SR code ####
    #############################################
    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image",
        default="inputs/user_upload"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/user_upload"
    )
    parser.add_argument(
        "--ddpm_steps",
        type=int,
        default=1000,
        help="number of ddpm sampling steps",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./stablesr_000117.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--vqgan_ckpt",
        type=str,
        default="./vqgan_cfw_00011.ckpt",
        help="path to checkpoint of VQGAN model",
    )  
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--dec_w",
        type=float,
        default=0.5,
        help="weight for combining VQGAN and Diffusion",
    )
    parser.add_argument(
        "--tile_overlap",
        type=int,
        default=32,
        help="tile overlap size (in latent)",
    )
    parser.add_argument(
        "--upscale",
        type=float,
        default=4.0,
        help="upsample scale",
    )
    parser.add_argument(
        "--colorfix_type",
        type=str,
        default="nofix",
        help="Color fix type to adjust the color of HR result according to LR input: adain (used in paper); wavelet; nofix",
    )
    parser.add_argument(
        "--vqgantile_stride",
        type=int,
        default=1000,
        help="the stride for tile operation before VQGAN decoder (in pixel)",
    )
    parser.add_argument(
        "--vqgantile_size",
        type=int,
        default=1280,
        help="the size for tile operation before VQGAN decoder (in pixel)",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=512,
        help="input size",
    )
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    return lp, op, pp, args

if __name__ == "__main__":
    lp, op, pp, args = parse_args()
    print("Optimizing " + args.model_path)    
    # Set up random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    seed_everything(args.seed)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)    
        
    train_proposed(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)    
    # All done 
    print("\nTraining complete.")
