import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import argparse
# 确保路径能找到 OGNIDC
from model.ognidc import OGNIDC

def get_ognidc_args():
    """封装你提供的 config.py 逻辑"""
    parser = argparse.ArgumentParser(description='OGNIDC')
    # Dataset
    parser.add_argument('--dir_data',
                        type=str,
                        default='../datasets/nyudepthv2_h5',
                        help='path to dataset')
    parser.add_argument('--train_data_name',
                        type=str,
                        default='NYU',
                        help='dataset name')
    parser.add_argument('--val_data_name',
                        type=str,
                        default='NYU',
                        choices=('NYU',
                                'NYU_FULL_RES',
                                'KITTIDC',
                                'VKITTI',
                                'VOID',
                                'BlendedMVS',
                                'TartanAir',
                                'DIODE_Indoor',
                                'DIODE_Outdoor',
                                'ETH3D_Indoor',
                                'ETH3D_Outdoor',
                                'ETH3D_SfM_Indoor',
                                'ETH3D_SfM_Outdoor',
                                'IRS',
                                'iBims',
                                'ARKitScenes',
                                'Uniformat',
                                ),
                        help='dataset name')
    parser.add_argument('--split_json',
                        type=str,
                        default='../data_json/nyu.json',
                        help='path to json file')
    parser.add_argument('--benchmark_gen_split',
                        type=str,
                        default='test',
                        help='generate test or val split')
    parser.add_argument('--benchmark_save_name',
                        type=str,
                        default='',
                        help='the name of the saved subset')
    parser.add_argument('--patch_height',
                        type=int,
                        default=228,
                        help='height of a patch to crop')
    parser.add_argument('--patch_width',
                        type=int,
                        default=304,
                        help='width of a patch to crop')
    parser.add_argument('--resize_height',
                        type=int,
                        default=240,
                        help='height to resize to')
    parser.add_argument('--resize_width',
                        type=int,
                        default=320,
                        help='width to resize to')
    parser.add_argument('--top_crop',
                        type=int,
                        default=0,
                        help='top crop size for KITTI dataset')
    parser.add_argument('--depth_scale_multiplier',
                        type=float,
                        default=1.0,
                        help='multiply input and gt depth both by this factor')
    parser.add_argument('--training_patch_size',
                        type=int,
                        default=-1,
                        help='height of a patch to crop')
    parser.add_argument('--data_normalize_median',
                        type=int,
                        default=True,
                        help='make the median of training data to be always 1')
    parser.add_argument('--mixed_dataset_total_length',
                        type=int,
                        default=-1,
                        help='the length of the mixed dataset; used to decide the length of each epoch')
    parser.add_argument('--precomputed_depth_data_path',
                        type=str,
                        default='',
                        help='path to json file')
    parser.add_argument('--precomputed_alignment_method',
                        type=str,
                        default='disparity',
                        help='path to json file')
    parser.add_argument('--load_dav2',
                        type=int,
                        default=False,
                        help='path to json file')

    # Hardware
    parser.add_argument('--seed',
                        type=int,
                        default=43,
                        help='random seed point')
    parser.add_argument('--gpus',
                        type=str,
                        default="0,1,2,3,4,5,6,7",
                        help='visible GPUs')
    parser.add_argument('--port',
                        type=str,
                        default='29500',
                        help='master port')
    parser.add_argument('--tcp_port',
                        type=int,
                        default=8080,
                        help='tcp port used for multiprocessing')
    parser.add_argument('--address',
                        type=str,
                        default='localhost',
                        help='master address')
    parser.add_argument('--num_threads',
                        type=int,
                        default=4,
                        help='number of threads')
    parser.add_argument('--multiprocessing',
                        action='store_true',
                        default=False,
                        help='do multiprocessing for DDP')

    # Network
    parser.add_argument('--model',
                        type=str,
                        default='OGNIDC',
                        choices=('OGNIDC',
                                ),
                        help='main model name')
    parser.add_argument('--integration_alpha',
                        type=float,
                        default=5.0,
                        help='relative weight of the depth term in optim layer')
    parser.add_argument('--spn_type',
                        type=str,
                        default='dyspn',
                        choices=['dyspn', 'nlspn'],
                        help='spn module to use. default: dyspn')
    parser.add_argument('--prop_time',
                        type=int,
                        default=6,
                        help='number of propagation. set to 0 if use no spn')
    parser.add_argument('--prop_kernel',
                        type=int,
                        default=3,
                        help='propagation kernel size')
    parser.add_argument('--preserve_input',
                        action='store_true',
                        default=False,
                        help='preserve input points by replacement')
    parser.add_argument('--affinity',
                        type=str,
                        default='TGASS',
                        choices=('AS', 'ASS', 'TC', 'TGASS'),
                        help='affinity type (dynamic pos-neg, dynamic pos, '
                            'static pos-neg, static pos, none')
    parser.add_argument('--affinity_gamma',
                        type=float,
                        default=0.5,
                        help='affinity gamma initial multiplier '
                            '(gamma = affinity_gamma * number of neighbors')
    parser.add_argument('--conf_prop',
                        action='store_true',
                        default=True,
                        help='confidence for propagation')
    parser.add_argument('--no_conf',
                        action='store_false',
                        dest='conf_prop',
                        help='no confidence for propagation')
    parser.add_argument('--pred_depth',
                        type=int,
                        default=False,
                        help='depth branch (kept for compatibility)')
    parser.add_argument('--pred_context_feature',
                        type=int,
                        default=True,
                        help='predict context feature for GRU')
    parser.add_argument('--pred_confidence_input',
                        type=int,
                        default=False,
                        help='pred confidence value for each px with gt observation. used for filtering KITTI inputs')
    parser.add_argument('--GRU_iters',
                        type=int,
                        default=1,
                        help='number of GRU iterations')
    parser.add_argument('--gru_context_dim',
                        type=int,
                        default=64,
                        help='the dimension of context feature')
    parser.add_argument('--gru_hidden_dim',
                        type=int,
                        default=64,
                        help='the dimension of hidden state')
    parser.add_argument('--conf_min',
                        type=float,
                        default=1.0,
                        help='min value of confidence of depth grad. default=1.0, i.e, conf disabled.')
    parser.add_argument('--optim_layer_scale_factor',
                        type=float,
                        default=1.0,
                        help='scale down the system for better numerical stability. default: 1.0')
    parser.add_argument('--optim_layer_input_clamp',
                        type=float,
                        default=1.0,
                        help='clamp for better numerical stablity')
    parser.add_argument('--backbone_mode',
                        type=str,
                        default='rgbd',
                        choices=('rgb', 'rgbd'),
                        help='rgb or rgbd input')
    parser.add_argument('--backbone',
                        type=str,
                        default='cformer',
                        choices=('cformer', ),
                        help='rgb or rgbd input')
    parser.add_argument('--training_depth_mask_out_rate',
                        type=float,
                        default=0.0,
                        help='masking out *some* depth pixels in this proportion of samples in the minibatch')
    parser.add_argument('--training_depth_mask_integ_depth',
                        type=int,
                        default=False,
                        help='also mask out the depth received by the integ layer')
    parser.add_argument('--training_depth_random_shift_range',
                        type=float,
                        default=0.0,
                        help='augment the training set depth by adding random shift')
    parser.add_argument('--depth_activation_format',
                        type=str,
                        default='exp',
                        choices=['exp', 'linear'],
                        help='integration in depth space or log-depth space')
    parser.add_argument('--backbone_output_downsample_rate',
                        type=int,
                        default=4,
                        help='the backbone output downsample rate')
    parser.add_argument('--depth_downsample_method',
                        type=str,
                        default='min',
                        choices=('mean', 'min'),
                        help='pooling method used to downsample depth')
    parser.add_argument('--whiten_sparse_depths',
                        type=int,
                        default=False,
                        help='make the median of sparse depth to be 1.')
    parser.add_argument('--gru_internal_whiten_method',
                        type=str,
                        default='mean',
                        choices=('mean', 'median'),
                        help='make the median of sparse depth to be 1.')

    # Training
    parser.add_argument('--loss',
                        type=str,
                        default='1.0*SeqL1+1.0*SeqL2',
                        help='loss function configuration')
    parser.add_argument('--laplace_loss_min_beta',
                        type=float,
                        default=-2.0,
                        help='clamp the beta value in laplace to this value')
    parser.add_argument('--gmloss_scales',
                        type=int,
                        default=4,
                        help='how many scales to use for the grad matching loss')
    parser.add_argument('--sequence_loss_decay',
                        type=float,
                        default=0.9,
                        help='sequence loss decay rate (see RAFT)')
    parser.add_argument('--intermediate_loss_weight',
                        type=float,
                        default=0.0,
                        help='add loss to the pred_init')
    parser.add_argument('--start_epoch',
                        type=int,
                        default=1,
                        help='epoch to start (used for resume)')
    parser.add_argument('--epochs',
                        type=int,
                        default=36,
                        help='number of epochs to train')
    parser.add_argument('--milestones',
                        nargs="+",
                        type=int,
                        default=[18, 24, 28, 43],
                        help='learning rate decay schedule')
    parser.add_argument('--opt_level',
                        type=str,
                        default='O0',
                        choices=('O0', 'O1', 'O2', 'O3'))
    parser.add_argument('--pretrain',
                        type=str,
                        default=None,
                        help='ckpt path')
    parser.add_argument('--resume',
                        action='store_true',
                        help='resume training')
    parser.add_argument('--test_only',
                        action='store_true',
                        help='test only flag')
    parser.add_argument('--batch_size',
                        type=int,
                        default=12,
                        help='input batch size PER GPU for training')
    parser.add_argument('--max_depth',
                        type=float,
                        default=10.0,
                        help='maximum depth. Loss and vis clamped by this value.')
    parser.add_argument('--augment',
                        type=bool,
                        default=True,
                        help='data augmentation')
    parser.add_argument('--no_augment',
                        action='store_false',
                        dest='augment',
                        help='no augmentation')
    parser.add_argument('--test_augment',
                        type=int,
                        default=False,
                        help='test time flip lr')
    parser.add_argument('--flip',
                        type=int,
                        default=True,
                        help='data augmentation w/ filp horizontal')
    parser.add_argument('--no_flip',
                        action='store_false',
                        dest='flip',
                        help='no flip')
    parser.add_argument('--random_rot_deg',
                        type=float,
                        default=0.0,
                        help='data augmentation w/ rotation')
    parser.add_argument('--train_depth_noise',
                        type=str,
                        default='0.0',
                        help='add random noise to training depth')
    parser.add_argument('--val_depth_noise',
                        type=str,
                        default='0.0',
                        help='add random noise to validation depth')
    parser.add_argument('--train_depth_pattern',
                        type=str,
                        default='500',
                        help='number of sparse samples')
    parser.add_argument('--train_sfm_max_dropout_rate',
                        type=float,
                        default=0.0,
                        help='number of sparse samples')
    parser.add_argument('--train_depth_velodyne_random_baseline',
                        type=int,
                        default=True,
                        help='number of sparse samples')
    parser.add_argument('--val_depth_pattern',
                        type=str,
                        default='500',
                        help='number of sparse samples')
    parser.add_argument('--inference_pattern_type',
                        type=str,
                        default='random',
                        help='pattern-specific inference')
    parser.add_argument('--num_pattern_types',
                        type=int,
                        default=3,
                        help='how many pattern types does the system support. Currently 3: random/velodyne/sfm')
    parser.add_argument('--backbone_pattern_condition_format',
                        type=str,
                        default='none',
                        choices=(
                            'none',
                            'block'
                        ),
                        help='how should the backbone be conditioned on depth pattern')
    parser.add_argument('--lidar_lines',
                        type=int,
                        default=64,
                        help='the extracted lidar lines')
    parser.add_argument('--test_crop',
                        action='store_true',
                        default=False,
                        help='crop the top for kitti test')
    parser.add_argument('--grad_clip',
                        type=float,
                        default=1.0,
                        help='clip the gradients')
    parser.add_argument('--grad_format',
                        type=str,
                        default='grad',
                        choices=('grad', 'normal'),
                        help='clip the gradients')

    # Summary
    parser.add_argument('--num_summary',
                        type=int,
                        default=4,
                        help='maximum number of summary images to save')

    # Optimizer
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='learning rate')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.5,
                        help='learning rate multiplicative factors')
    parser.add_argument('--optimizer',
                        default='ADAMW',
                        choices=('SGD', 'ADAM', 'ADAMW', 'RMSPROP'),
                        help='optimizer to use (SGD | ADAM | RMSprop | ADAMW)')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        help='SGD momentum')
    parser.add_argument('--betas',
                        type=tuple,
                        default=(0.9, 0.999),
                        help='ADAM | ADAMW beta')
    parser.add_argument('--epsilon',
                        type=float,
                        default=1e-8,
                        help='ADAM | ADAMW epsilon for numerical stability')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.01,
                        help='weight decay')
    parser.add_argument('--warm_up',
                        action='store_true',
                        default=True,
                        help='do lr warm up during the 1st epoch')
    parser.add_argument('--no_warm_up',
                        action='store_false',
                        dest='warm_up',
                        help='no lr warm up')
    parser.add_argument('--num_resolution',
                        type=int,
                        default=1,
                        help='changing the resolution')
    parser.add_argument('--multi_resolution_learnable_input_weights',
                        type=int,
                        default=0,
                        help='when using multires, have learnable input weights')
    parser.add_argument('--multi_resolution_learnable_gradients_weights',
                        type=str,
                        default='learnable',
                        help='when using multires, have learnable gradients weights')

    # Logs
    parser.add_argument('--log_dir',
                        type=str,
                        default='../experiments/',
                        help='dir for log')
    parser.add_argument('--print_freq',
                        type=int,
                        default=1,
                        help='print frequency of tqdm')
    parser.add_argument('--save',
                        type=str,
                        default='trial',
                        help='file name to save')
    parser.add_argument('--save_full',
                        action='store_true',
                        default=False,
                        help='save optimizer, scheduler and amp in '
                            'checkpoints (large memory)')
    parser.add_argument('--save_result_only',
                        action='store_true',
                        default=False,
                        help='save result images only with submission format')
    parser.add_argument('--save_pointcloud_visualization',
                        action='store_true',
                        default=False,
                        help='save pointcloud in ply format')
    parser.add_argument('--save_uniformat_max_dataset_length',
                        type=int,
                        default=800,
                        help='if the dataset is too long, subsample to this length')
    args, _ = parser.parse_known_args([])
    return args

class DepthEngine:
    def __init__(self, checkpoint_path, max_depth=300.0, num_resolution=3):
        # 1. 获取包含所有默认参数的 args 对象
        self.args = get_ognidc_args()
        
        # 2. 覆盖推理所需的关键参数
        self.args.pretrain = checkpoint_path
        self.args.max_depth = max_depth
        self.args.num_resolution = num_resolution
        self.args.test_only = True
        self.args.backbone_mode = 'rgbd'
        self.args.data_normalize_median = 1
        self.args.multi_resolution_learnable_gradients_weights = "uniform"
        self.args.load_dav2 = 1
        self.args.GRU_iters = 1
        self.args.optim_layer_input_clamp = 1.0
        self.args.depth_activation_format = 'exp'
        self.args.whiten_sparse_depths = 1
        self.args.pred_confidence_input = 1

        self.net = OGNIDC.from_pretrained("zuoym15/OMNI-DC", args=self.args).eval()
        self.net = nn.DataParallel(self.net).cuda()
        
        self.t_rgb = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.diviser = int(4 * 2 ** (self.args.num_resolution - 1))

    def generate(self, image_names, rgb_dir, dep_dir, output_dir, step_suffix):
        os.makedirs(output_dir, exist_ok=True)
        for name in image_names:
            rgb_path = os.path.join(rgb_dir, f"{name}_{step_suffix}.png")
            sparse_dep_path = os.path.join(dep_dir, f"{name}.npy")
            
            if not os.path.exists(rgb_path): continue

            rgb = self.t_rgb(Image.open(rgb_path)).unsqueeze(0).cuda()
            dep_sp = torch.from_numpy(np.load(sparse_dep_path)).float().unsqueeze(0).unsqueeze(0).cuda()
            
            # Padding 逻辑
            _, _, H, W = rgb.shape
            H_pad = (self.diviser - H % self.diviser) % self.diviser
            W_pad = (self.diviser - W % self.diviser) % self.diviser
            rgb_in = torch.nn.functional.pad(rgb, (0, W_pad, 0, H_pad))
            dep_in = torch.nn.functional.pad(dep_sp, (0, W_pad, 0, H_pad))

            with torch.no_grad():
                output = self.net({'rgb': rgb_in, 'dep': dep_in, 
                                   'K': torch.eye(3).reshape(1, 3, 3).cuda(), 'pattern': 0})

            depth_pred = output['pred'][..., :H, :W].squeeze().cpu().numpy()
            
            # 【修复 NpzFile 报错】：必须使用 np.save 而不是 torch.save
            out_name = os.path.join(output_dir, f"{name}_{step_suffix}.npy")
            np.save(out_name, depth_pred)