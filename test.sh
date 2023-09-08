#!/bin/bash
### make sure that you have modified the EXP_NAME, CKPT, DATASETS_TEST
eval "$(conda shell.bash hook)"
conda activate dire

EXP_NAME="models"
CKPT="lsun_adm.pth"
DATASETS_TEST="dire_images/test/imagenet/"
python test.py --gpus 0 --ckpt $CKPT --exp_name $EXP_NAME datasets_test $DATASETS_TEST