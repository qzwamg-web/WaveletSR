<p align="center">

  <h1 align="center">WaveletSR environment</h1>

  <h2 align="center">NTIRE 2026 (CVPR 2026 Workshop)</h2>

  <h3 align="center">Environment Setup</h3>
  <div align="center"></div>
</p>

## Image Rendering and Generation

This section describes the procedure to synthesize images from the trained Gaussian Splatting models.

### 1. Data Preparation
For each scene, ensure the directory structure is organized as follows. You need to manually prepare the `sparse/0/` directory and the `cfg_args` file.

- **Directory Structure:**
  ```text
  <scene_root>/
  ├── sparse/
  │   └── 0/
  │       ├── images.bin
  │       └── cameras.bin
  ├── point_cloud/          # Generated after training
  │   └── iteration_30000/
  └── cfg_args              # Configuration file
Configuration File (cfg_args):
Create a new file named cfg_args in the scene root directory. Copy and paste the following content, ensuring the model_path and source_path match your local directory:

```python
Namespace(data_device='cuda', eval=False, images='images', depths='', model_path='datasets/WestAccommodationAreas', resolution=-1, sh_degree=3, source_path='datasets/WestAccommodationAreas', white_background=False, train_test_exp=False)
```
[!IMPORTANT]
The cfg_args, sparse/, and point_cloud/ directories must be located in the same parent folder for the renderer to correctly load the parameters.

### 2. Execution
Navigate to the gaussian-splatting workspace and run the following command to generate the final images:

### Example for EastResearchAreas scene
```bash
GT_CAMERA_INTER_NUM=0 python render.py -m track2/EastResearchAreas
```
3. Output
The rendered images will be saved in:
[model_path]/train/render

We have provided sample rendered images in the accompanying submission package for verification.


## Installation

### Step 1. Create a new conda environment

```bash
conda create -y -n wavelet python=3.8
conda activate wavelet
```

### Step 2. Install PyTorch with CUDA 11.8
```bash
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

### Step 3. Install Python dependencies
Then install all dependencies by:
```bash
pip install -r requirements.txt
```

### Step 4. Install local packages
```bash
cd third_parties
# git clone https://github.com/CompVis/taming-transformers.git
cd taming-transformers
pip install -e .
cd ..
```
```bash
# git clone https://github.com/openai/CLIP.git
cd CLIP
pip install -e .
cd ..
```

```bash
# git clone https://github.com/openai/CLIP.git
cd fused-ssim
pip install -e . --no-build-isolation
cd ..
```

```bash
pip install -e .
cd ..
```

### Step 5. Install CUDA/C++ extensions
```bash
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

### Step 6. Model download

Please download the following pretrained checkpoints and place them in the `third_parties/weights` directory.

- **StableSR-Turbo**: [stablesr_turbo.ckpt](https://huggingface.co/Iceclear/StableSR/resolve/main/stablesr_turbo.ckpt)
- **VQGAN autoencoder weights**: [vqgan_cfw_00011.ckpt](https://huggingface.co/Iceclear/StableSR/resolve/main/vqgan_cfw_00011.ckpt)

The model weight folder should be organized as follows:

```bash
WaveletSR/
  └── third_parties
       └── weights
            ├── stablesr_turbo.ckpt
            └── vqgan_cfw_00011.ckpt
```

### Step 7. Fix the PyTorch Lightning import issue

In `WaveletSR/third_parties/ldm/models/diffusion/ddpm.py`, change line 19 from

```python
from pytorch_lightning.utilities.distributed import rank_zero_only
```
to
```python
from pytorch_lightning.utilities.rank_zero import rank_zero_only
```


## Model Training

### 1. Preparation
Place the rendered `images/` folder in the same parent directory as the `point_cloud/` folder. Ensure the data structure aligns with the 3DGS requirements.

### 2. Execution
Navigate to the `ntire/` directory and run the training script:

```bash
bash ntire.sh
```
### 3. Configuration
You can switch scenes or adjust hyperparameters by modifying the following variables within ntire.sh.

Key Parameters:

scenes: Specify the scene name (e.g., "NorthAreas" or "EastResearchAreas").

tracks: The challenge track (e.g., "track2").

points: Maximum number of Gaussians.

For EastResearchAreas, set to 4500000.

For NorthAreas, set to 6000000.

gpu: Designated GPU ID for training.

GS_iters: Total training iterations (default: 30000).


### Example configuration in ntire.sh
```
scenes=("NorthAreas") 
tracks="track2"
points=6000000 
dataset_path="test/${tracks}"
pretrained_base_path="test/${tracks}" 
gpu=3
HR_factor=5
GS_iters=30000
```
## Model Rendering
To render the final point cloud and obtain the results, use the answer.py script. This script requires the path to the trained model and the original camera parameters.

Execution Command
```
python answer.py \
  -m /path/to/your/outputs/NorthAreas_track1 --ext_cam_path /path/to/your/test/track1/NorthAreas
```
Argument Descriptions:

-m: The directory path containing the output point cloud (.ply and associated parameters).

--ext_cam_path: The directory path containing the original camera extrinsic and intrinsic files.

Note: Ensure that the paths provided to -m and --ext_cam_path are absolute paths or correctly relative to your current working directory.




## Acknowledgements
This project is built upon [Improved_GS](https://github.com/XiaoBin2001/Improved-GS.git), [3DSR](https://github.com/Consistent3DSR/3DSR.git) and [StableSR](https://github.com/IceClear/StableSR.git). Please follow the license of MipSplatting and StableSR. We thank all the authors for their great work and repos.


The contents of the file `utils/loss_utils.py` are based on publicly available code authored by Evan Su, which falls under the permissive MIT license.
**Title:** pytorch-ssim  
**Project code:** [https://github.com/Po-Hsun-Su/pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim) Copyright Evan Su, 2017  
**License:** [https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/LICENSE.txt](https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/LICENSE.txt) (MIT)