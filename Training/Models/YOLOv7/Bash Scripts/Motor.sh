#!/bin/bash
##SBATCH -w gnode076
#SBATCH -A aryamaan_roy
#SBATCH -c 10
##SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
##SBATCH --mail-type=ALL
##SBATCH --mem-per-cpu=3000
#SBATCH --output=Motor.txt

source activate base

module load u18/cuda/10.2
module load u18/cudnn/7.6.5-cuda-10.2

export CUDA_VISIBLE_DEVICES=0

cd /home2/aryamaan_roy/Motorcycle-Violations-Complete/Training/YOLOv7/yolov7_motor

python3 train.py --weights ./../weights/pretrained.pt --data "data/motor.yaml" --workers 2 --batch-size 4 --cfg cfg/training/yolov7.yaml --name yolov7 --hyp data/hyp.scratch.p5.yaml --epochs 150

echo ----Training Complete----
