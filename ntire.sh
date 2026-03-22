# #!/bin/bash

scenes=("EastResearchAreas") # "WestResearchAreas"  "WestTeachingAreas" WestAccommodationAreas EastResearchAreas NorthAreas
tracks="track1"
points=4500000


# 数据集根目录路径
dataset_path="/gdata/cold1/fujiaye/qzwang/test/${tracks}"
# 预训练模型根目录 
pretrained_base_path="/gdata/cold1/fujiaye/qzwang/test/${tracks}" 

# 公共参数
gpu=1
HR_factor=5
GS_iters=30000
export NUM_ITERS=${GS_iters}

# ================= 循环执行 =================
for scene in "${scenes[@]}"; do
    echo "=================================================="
    echo "🚀 开始场景训练: ${scene}"
    
    # 动态路径定义
    exp_dir="./outputs/${scene}_${tracks}"
    mkdir -p ${exp_dir}
    
    # 确保每个场景都有对应的预训练模型目录
    current_pretrained_path="${pretrained_base_path}/${scene}"

    echo "加载预训练点云自: ${current_pretrained_path}"

    # 执行训练命令
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$gpu python train_sr.py \
        -s "${dataset_path}/${scene}" \
        -m "${current_pretrained_path}" \
        -r ${HR_factor} \
        --port $((6000 + gpu)) \
        --output_folder "${exp_dir}" \
        --load_pretrain \
        --config ./configs/stableSRNew/v2-finetune_text_T_512.yaml \
        --ckpt ./third_parties/weights/stablesr_turbo.ckpt \
        --init-img "${dataset_path}/${scene}/images" \
        --outdir "${exp_dir}" \
        --ddpm_steps 5 \
        --dec_w 0.3 \
        --seed 42 \
        --n_samples 2 \
        --vqgan_ckpt ./third_parties/weights/vqgan_cfw_00011.ckpt \
        --colorfix_type wavelet \
        --upscale 4 \
        --fidelity_train_en \
        --wt_lr 1 \
        --densify_end $((NUM_ITERS / 2)) \
        --budget ${points}

    echo "--------------------------------------------------"
done