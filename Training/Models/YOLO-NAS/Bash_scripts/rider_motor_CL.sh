#!/bin/bash
##SBATCH -w gnode076
#SBATCH -A aryamaan_roy
#SBATCH -c 10
##SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
##SBATCH --mail-type=ALL
##SBATCH --mem-per-cpu=3000
#SBATCH --output=rider_motor_CL.txt

source activate base

module load u18/cuda/10.2
module load u18/cudnn/7.6.5-cuda-10.2

export CUDA_VISIBLE_DEVICES=0

cd /home2/aryamaan_roy/Motorcycle-Violations-Complete/Training/YOLO-NAS

python3 train_med_rider_motor_CL.py

echo ----Training Complete----
