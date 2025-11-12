#!/bin/bash

# Simple run script for PLM-CUP

# Parse command line arguments
MODEL=""
DATA_PATH=""
PRETRAIN_PATH=""
PRETRAIN_MODEL=""
SAVE_PATH=""
BATCH_SIZE=64
EPOCHS=500
LEARNING_RATE=0.0005
DEVICE="cuda:0"
GPT_LAYERS=6
USE_LORA="True"
TRAIN_RATIO=100
IS_TRANSFER="False"
LOAD_MODEL=""
SEED=42

while [ $# -gt 0 ]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --data)
            DATA_PATH="$2"
            shift 2
            ;;
        --pretrain_path)
            PRETRAIN_PATH="$2"
            shift 2
            ;;
        --pretrain_model)
            PRETRAIN_MODEL="$2"
            shift 2
            ;;
        --save)
            SAVE_PATH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --gpt_layers)
            GPT_LAYERS="$2"
            shift 2
            ;;
        --use_lora)
            USE_LORA="$2"
            shift 2
            ;;
        --train_ratio)
            TRAIN_RATIO="$2"
            shift 2
            ;;
        --is_transfer)
            IS_TRANSFER="$2"
            shift 2
            ;;
        --load_model)
            LOAD_MODEL="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required parameters
if [ -z "$MODEL" ] || [ -z "$DATA_PATH" ] || [ -z "$PRETRAIN_PATH" ]; then
    echo "Usage: ./run.sh --model MODEL_NAME --data /full/path/to/data --pretrain_path /full/path/to/pretrained/model"
    echo ""
    echo "Required parameters:"
    echo "  --model         Model name (e.g., PLM_CUP)"
    echo "  --data          Full path to dataset"
    echo "  --pretrain_path Full path to pre-trained model (GPT-2 or Qwen3)"
    echo ""
    echo "Optional parameters:"
    echo "  --save          Full path to save model"
    echo "  --load_model    Full path to load pre-trained model (for transfer)"
    echo "  --batch_size    Batch size (default: 64)"
    echo "  --epochs        Number of epochs (default: 500)"
    echo "  --learning_rate Learning rate (default: 0.0005)"
    echo "  --device        Device (default: cuda:0)"
    echo "  --gpt_layers    Number of GPT layers (default: 6)"
    echo "  --use_lora      Use LoRA (default: True)"
    echo "  --train_ratio   Percentage of training data (default: 100)"
    echo "  --is_transfer   Enable transfer learning (default: False)"
    exit 1
fi

# Build command
CMD="python main.py"
CMD="$CMD --model $MODEL"
CMD="$CMD --data $DATA_PATH"
CMD="$CMD --gpt_path $PRETRAIN_PATH"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --learning_rate $LEARNING_RATE"
CMD="$CMD --device $DEVICE"
CMD="$CMD --gpt_layers $GPT_LAYERS"
CMD="$CMD --use_lora $USE_LORA"
CMD="$CMD --train_ratio $TRAIN_RATIO"
CMD="$CMD --is_transfer $IS_TRANSFER"

if [ -n "$SEED" ]; then
    CMD="$CMD --seed $SEED"
fi

# Add optional parameters
if [ -n "$SAVE_PATH" ]; then
    CMD="$CMD --save $SAVE_PATH"
fi

if [ -n "$LOAD_MODEL" ]; then
    CMD="$CMD --load_model $LOAD_MODEL"
fi

# Additional fixed parameters
CMD="$CMD --input_dim 3"
CMD="$CMD --channels 64"
CMD="$CMD --num_nodes 225"
CMD="$CMD --input_len 6"
CMD="$CMD --output_len 1"
CMD="$CMD --dropout 0.1"
CMD="$CMD --weight_decay 0.0001"
CMD="$CMD --print_every 50"
CMD="$CMD --es_patience 100"
CMD="$CMD --lr_decay 0.2"
CMD="$CMD --lr_decay_patience 30"
CMD="$CMD --min_learning_rate 0.00005"
CMD="$CMD --lora_r 16"
CMD="$CMD --lora_alpha 64"
CMD="$CMD --pretrain_model $PRETRAIN_MODEL"
CMD="$CMD --U 0"

# Run the command
echo "Running PLM-CUP with configuration:"
echo "  Model: $MODEL"
echo "  Data: $DATA_PATH"
echo "  Pre-trained: $PRETRAIN_PATH"
echo "  Device: $DEVICE"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo ""
echo "Executing command:"
echo "$CMD"
echo ""

eval $CMD