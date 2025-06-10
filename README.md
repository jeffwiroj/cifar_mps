# CIFAR MPS Project

## 📁 Project Structure

```
root_dir/
├── cifar_mps/   # Main Package
├── examples/     # Scripts to run experiments or utilities
```

## Overview

This project implements CIFAR image classification. Currently supports training resnet18 and smol_resnet.
TODO: Adding in SimSiam, VicReg, and possibly mixmatch

## Installation

This project uses uv 

```bash
# Install uv if you haven't already
# Install dependencies
uv sync
```

## Usage

### Training Supervised Experiments
See train.py args to see what you can configure. To use wandb, turn on --use-wandb, make sure you save your api key somewhere handy, it will be prompted during the initial run.
```bash
uv run examples/train.py --model resnet18 
```

## Supervised Benchmark Results on Apple M1
- All experiments should take around 6 minutes to run
- The reported accuracy is on the validation dataset, here we take cifar's test set as the val, and don't actually have a test set.
- For consistency, we are fixing batch size = 512 and epochs = 12
- sgd refers to sgd with momentum = 0.9 and no nesterov

| Model | Batch Size | Epochs | Optimizer | Scheduler | Augmentations | LR Warmup | Learning Rate | Weight Decay | Accuracy |
|-------|------------|--------|-----------|-----------|---------------|-----------|---------------|--------------|----------|
| smol_resnet | 512 | 12 | adamw | cosine | basic | 0.3 | 0.05 | 0.05 | 86.6% | 
| smol_resnet | 512 | 12 | sgd | cosine | basic | 0.3 | 0.1 | 0.05 | 85.5% |




<!-- ## Project Components

- **`cifar_mps/`** - Core implementation of all cifar related things
- **`scripts/`** - Training scripts and experiment utilities   -->