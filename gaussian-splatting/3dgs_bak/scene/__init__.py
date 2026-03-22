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

import os
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    @staticmethod
    def _orthogonalize_rotation(R):
        U, _, Vt = np.linalg.svd(R)
        R_ortho = U @ Vt
        if np.linalg.det(R_ortho) < 0:
            U[:, -1] *= -1
            R_ortho = U @ Vt
        return R_ortho.astype(np.float32)

    def _interpolate_test_cameras(self, cameras, num_between, train_test_exp):
        if num_between <= 0 or len(cameras) < 2:
            return cameras

        interpolated = []
        next_uid = max((cam.uid for cam in cameras), default=-1) + 1

        for idx in range(len(cameras) - 1):
            cam_a = cameras[idx]
            cam_b = cameras[idx + 1]
            interpolated.append(cam_a)

            R_a = np.asarray(cam_a.R, dtype=np.float32)
            R_b = np.asarray(cam_b.R, dtype=np.float32)
            T_a = np.asarray(cam_a.T, dtype=np.float32)
            T_b = np.asarray(cam_b.T, dtype=np.float32)

            for j in range(num_between):
                t = float(j + 1) / float(num_between + 1)
                R_interp = self._orthogonalize_rotation((1.0 - t) * R_a + t * R_b)
                T_interp = ((1.0 - t) * T_a + t * T_b).astype(np.float32)
                fovx_interp = float((1.0 - t) * cam_a.FoVx + t * cam_b.FoVx)
                fovy_interp = float((1.0 - t) * cam_a.FoVy + t * cam_b.FoVy)

                interp_cam = Camera(
                    resolution=(cam_a.image_width, cam_a.image_height),
                    colmap_id=next_uid,
                    R=R_interp,
                    T=T_interp,
                    FoVx=fovx_interp,
                    FoVy=fovy_interp,
                    depth_params=None,
                    image=None,
                    invdepthmap=None,
                    image_name=f"{cam_a.image_name}_interp_{j + 1}_{cam_b.image_name}",
                    uid=next_uid,
                    data_device=str(cam_a.data_device),
                    train_test_exp=train_test_exp,
                    is_test_dataset=True,
                    is_test_view=True,
                )
                interpolated.append(interp_cam)
                next_uid += 1

        interpolated.append(cameras[-1])
        return interpolated

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.test_camera_interp_num = int(getattr(args, "test_camera_interp_num", os.getenv("GS_TEST_CAMERA_INTERP_NUM", 0)))

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print(f"Training cam number is {len(self.train_cameras[resolution_scale])}")
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)
            self.test_cameras[resolution_scale] = self._interpolate_test_cameras(
                self.test_cameras[resolution_scale],
                self.test_camera_interp_num,
                args.train_test_exp,
            )
            print(f"Test cam number is {len(self.test_cameras[resolution_scale])}")

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
