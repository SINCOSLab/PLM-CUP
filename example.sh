#!/bin/bash

sh run.sh \
  --model PLM_CUP \
  --data /gemini/data-1/QY/NJ \
  --pretrain_path /gemini/code/pstxm/pretrain/qwen3-0.6b \
  --pretrain_model qwen3-0.6b \
  --save /gemini/code/pstxm/logs \
  --batch_size 64 \
  --epochs 500 \
  --learning_rate 0.0005 \
  --device cuda:0 \
  --gpt_layers 6 \
  --use_lora True \
  --train_ratio 100 \
  --is_transfer False