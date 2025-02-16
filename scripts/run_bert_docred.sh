#!/bin/bash

# Check if the first argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <CUDA_VISIBLE_DEVICES>"
  exit 1
fi

# Assign the first argument to CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$1

# Specify the path to your pre-trained BERT model
# Please replace {Your pre-trained BERT path} with the actual path
MODEL_NAME_OR_PATH="{Your pre-trained BERT path}"

# Run the training script with the specified arguments
nohup python -u train.py --norm_by_sample \
--device 0 \
--model_name_or_path "$MODEL_NAME_OR_PATH" \
--WANDB_MODE disabled \
--num_pretrain_epochs 5.0 \
--num_train_epochs 30.0 \
--negative_sampling_ratio 1.0 \
--part_train_file ./dataset/docred/train_annotated.json \
--dev_file ./dataset/docred/dev_revised.json \
--test_file ./dataset/docred/test_revised.json \
--ratio_pos 1.0 \
--ratio_neg 1.0 \
--EMA_lambda 0.999 \
--diagnose_mode rule \
--indicator prob \
--sampling_mode best \
--test_batch_size 8 \
--minC 0.85 \
--transformer_type bert \
--eta 0.3 \
--gamma 50.0 > ./logs/DocRED_bert.log 2>&1 &