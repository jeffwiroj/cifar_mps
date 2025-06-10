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


<!-- ## Project Components

- **`cifar_mps/`** - Core implementation of all cifar related things
- **`scripts/`** - Training scripts and experiment utilities   -->