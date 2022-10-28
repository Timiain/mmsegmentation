#!/bin/sh
# 加载 anaconda
module load anaconda/2020.11
# 加载 cuda 11.3
module load cuda/11.3
# 激活 python 虚拟环境
source activate rs-mmseg
export PYTHONUNBUFFERED=1

./tools/dist_train.sh ./configs/swim_irs/tta/upernet_swin_tiny_patch4_window7_512x512_80k_vaihingen_pretrain_224x224_1K.py 2 --deterministic --seed 42