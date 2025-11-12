<div align="center">
  <img src="logo.png" alt="PLM-CUP Logo" width="100" height="100">
  
# PLM-CUP

  Exploiting Pre-trained Language Model for Cross-city Urban Flow Prediction
  Guided by Information-theoretic Analysis
</div>

---

## Environment Setup

### Option 1: Using GPT-2 as Pre-trained Model

Create conda environment from the provided configuration:

```bash
conda env create -f gpt2/environment_gpt2.yml
```

### Option 2: Using Qwen3-0.6B as Pre-trained Model

Create conda environment from the provided configuration:

```bash
conda env create -f qwen3/environment_qwen3.yml
```

## Installation

If you prefer manual installation instead of using the provided conda environments:

```bash
pip install torch transformers numpy pandas scipy
```

## Usage

After setting up the environment and extracting the pre-trained model, run the following commands:

```bash
# Basic training with GPT-2
sh run.sh --model PLM_CUP \
  --data /path/to/your/data \
  --pretrain_path /path/to/your/pretrain/directory/gpt2 \
  --pretrain_model gpt2

# Basic training with Qwen3-0.6B
sh run.sh --model PLM_CUP \
  --data /path/to/your/data \
  --pretrain_path /path/to/your/pretrain/directory/qwen3 \
  --pretrain_model qwen3-0.6b

# Transfer Learning
sh run.sh --model PLM_CUP \
  --data /path/to/your/data \
  --pretrain_path /path/to/your/pretrain/directory/[gpt2|qwen3] \
  --pretrain_model [gpt2|qwen3-0.6b] \
  --load_model /path/to/your/component \
  --is_transfer True \
  --train_ratio 100
```

**Note**: Replace `/path/to/your/pretrain/directory` with the actual path where you extracted the model files.

## Parameters

### Required Parameters

- `--model`: Model name (e.g., PLM_CUP)
- `--data`: Full path to dataset
- `--pretrain_path`: Full path to pre-trained model (GPT-2 or Qwen3)

### Optional Parameters

- `--save`: Full path to save model
- `--load_model`: Full path to load pre-trained model (for transfer)
- `--batch_size`: Batch size (default: 64)
- `--epochs`: Number of epochs (default: 500)
- `--learning_rate`: Learning rate (default: 0.0005)
- `--device`: Device (default: cuda:0)
- `--gpt_layers`: Number of GPT layers (default: 6)
- `--use_lora`: Use LoRA (default: True)
- `--train_ratio`: Percentage of training data (default: 100)
- `--is_transfer`: Enable transfer learning (default: False)
- `--pretrain_model`: Pre-trained model name (gpt2 or qwen3-0.6b)
- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha (default: 64)
- `--seed`: Random seed (default: 42)

## Dataset Format

- 15×15 grid
- 1-hour intervals
- 6 hours input → 1 hour output
