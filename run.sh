#!/bin/bash

#CUDA_VISIBLE_DEVICES=0 python main.py \
#    --exp_name CAIN_train \
#    --dataset vimeo90k \
#    --batch_size 16 \
#    --test_batch_size 16 \
#    --model cain \
#    --depth 3 \
#    --loss 1*L1 \
#    --max_epoch 200 \
#    --lr 0.0002 \
#    --log_iter 100 \
#    --mode test

#python main.py \
#    --exp_name Elegance_Denoised_Shifted_Scale4 \
#    --dataset elegance \
#    --batch_size 4 \
#    --test_batch_size 16 \
#    --model cain \
#    --depth 3 \
#    --loss 1*L1 \
#    --max_epoch 200 \
#    --lr 0.0002 \
#    --log_iter 100 \
#    --denoise \
#    --shift \

#python main.py \
#    --exp_name Elegance_Raw_Scale4 \
#    --dataset elegance \
#    --batch_size 4 \
#    --test_batch_size 16 \
#    --model cain \
#    --depth 3 \
#    --loss 1*L1 \
#    --max_epoch 200 \
#    --lr 0.0002 \
#    --log_iter 100 \

#python main.py \
#    --exp_name Elegance_Denoised_Scale4 \
#    --dataset elegance \
#    --batch_size 4 \
#    --test_batch_size 16 \
#    --model cain \
#    --depth 3 \
#    --loss 1*L1 \
#    --max_epoch 200 \
#    --lr 0.0002 \
#    --log_iter 100 \
#    --denoise \

python main.py \
    --exp_name Elegance_Shifted_Scale4 \
    --dataset elegance \
    --batch_size 4 \
    --test_batch_size 16 \
    --model cain \
    --depth 3 \
    --loss 1*L1 \
    --max_epoch 200 \
    --lr 0.0002 \
    --log_iter 100 \
    --shift \