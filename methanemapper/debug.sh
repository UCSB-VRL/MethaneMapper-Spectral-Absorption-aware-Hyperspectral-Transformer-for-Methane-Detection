#!/bin/bash

python hyper_main.py --masks --hyper_path ./data --dataset_file hyper_seg --output_dir ./exps/segm_model --batch_size 1 --num_workers 0 --frozen_weights ./exps/box_model/mAPfix_checkpoint.pth --epoch 100 --lr_drop 15
