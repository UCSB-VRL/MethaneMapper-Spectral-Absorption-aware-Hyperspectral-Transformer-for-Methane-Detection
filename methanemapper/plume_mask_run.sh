#! /bin/bash

python hyper_main.py --masks --hyper_path ./data --dataset_file hyper_seg --output_dir ./exps/segm_model --batch_size 12 --num_workers 8 --frozen_weights ./exps/box_model/checkpoint.pth --epoch 100 --lr_drop 15 --use_wandb True
