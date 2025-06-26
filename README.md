# PCEDNet Training and Prediction Guide

This repository contains tools for training and predicting with PCEDNet models on point cloud data.

## Training

**Train (remember to change path in train_pcednet.py + scales + descriptor):**
By default it accept: 
TARGET_SCALES = 16 # Max 16, must be power of 2 to match PCEDNet architecture
TARGET_FEATURES = 20 # Max 20
BATCHES = 512
EPOCHS = 50
FRACTION = 1 #0.1 (train only 10%) or 1 (train all)
USE_VALIDATION = False

python /home/cle/Work/ABC-Challenge/ABC-Project/IMLPointNet/train_pcednet.py

The predict file should be save in
path/
--ply/your.ply
--lb/your.lb
--SSM_Challenge-ABC/your.ssm

## Prediction

**Predict (remember to choose correct model):**
python /home/cle/Work/ABC-Challenge/ABC-Project/IMLPointNet/pred_pcednet.py \
    --model "/home/cle/scan24_IML_pcednet_16s_20f_512b_50e_lr01.h5" \
    --input "/home/cle/data/dtu_results_pc/IML_scan24/SSM_Challenge-ABC/**scan24**.ssm" \
    --output "/home/cle/data/dtu_results_pc/IML_scan24/feedback_**1**.ply" \
    --batch_size 2048 \
    --threshold 0.5 \
    --scales 16 \

## Utility Scripts

ano2lb.py -> convert annotation file from RollingDot tool to .lb file in ABC dataset format

ply2ABCply.py -> convert ply file from 2DGS->pointcloud to .ply in ABC dataset format

visualizer.py -> view result