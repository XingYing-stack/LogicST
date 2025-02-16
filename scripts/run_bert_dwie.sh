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
--data_dir ./dataset/dwie \
--model_name_or_path "$MODEL_NAME_OR_PATH" \
--num_pretrain_epochs 15.0 \
--num_train_epochs 30.0 \
--part_train_file ./dataset/dwie/train_incomplete_$2.json \
--dev_file ./dataset/dwie/dev.json \
--test_file ./dataset/dwie/test.json \
--negative_sampling_ratio 1.0 \
--rule_path ./mined_rules/rule_dwie.pl \
--num_class 66 \
--ratio_pos 1.0 \
--ratio_neg 1.0 \
--EMA_lambda 0.9995 \
--diagnose_mode rule \
--indicator prob \
--sampling_mode best \
--test_batch_size 8 \
--minC 0.9 \
--transformer_type bert \
--eta 0.3 \
--gamma 100.0 > ./logs/DWIE_bert.log 2>&1 &