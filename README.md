# CIFAR MPS Project

## 📁 Project Structure

```
root_dir/
├── src/ 
    ├── cifar_mps/   # Supervised Learning package
├── scripts/     # Scripts to run experiments or utilities
```

## Overview

This project implements CIFAR image classification. Currently supports training resnet18 and smol_resnet.
TODO: Will be adding fix-match, Sim-Siam, for semi/self supervised things. 

## Installation

This project uses uv 

```bash
# Install uv if you haven't already
# Install dependencies
uv sync
```

## Usage

### Training Supervised Experiments
See train.py args to see what you can configure.
```bash
uv run scripts/train.py --model resnet18 
```


<!-- ## Project Components

- **`cifar_mps/`** - Core implementation of all cifar related things
- **`scripts/`** - Training scripts and experiment utilities   -->