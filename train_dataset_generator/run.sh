#! /bin/bash

#python read_data.py -cols 10
python train_data_generator.py -snfmf column_wise -sensors 8 -pxl 100000
