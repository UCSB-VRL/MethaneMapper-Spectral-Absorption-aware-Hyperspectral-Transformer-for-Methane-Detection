#!/bin/bash

python hyper_main.py --hyper_path ./data --dataset_file hyper_seg --pretrained True --output_dir ./exps/box_model --batch_size 12 --num_workers 8
#CUDA_VISIBLE_DEVICES=2 python hyper_main.py --masks --hyper_path ./data --dataset_file hyper_seg --output_dir ./exps/segm_model --batch_size 12 --num_workers 6 --frozen_weights ./exps/box_model/mAPfix_checkpoint.pth --epoch 100 --lr_drop 15 --use_wandb True
#CUDA_VISIBLE_DEVICES=2 python hyper_main.py --masks --hyper_path ./data --dataset_file hyper_seg --output_dir ./exps/segm_model --batch_size 12 --num_workers 6 --frozen_weights ./exps/box_model/mAPfix_checkpoint.pth --epoch 100 --lr_drop 15
