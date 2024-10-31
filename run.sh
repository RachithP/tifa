#!/bin/bash

#SBATCH -p g24
#SBATCH --gres=gpu:4
#SBATCH -c 16

source /home/rachithp/miniforge3/bin/activate tifa

python run_scoring.py --imgs_file /home/rachithp/code/tifa/image_generation/sd21_base_fp16_ov/imgs.json --output_file /home/rachithp/code/tifa/color.csv
