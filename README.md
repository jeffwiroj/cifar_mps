# CIFAR MPS Project

## 📁 Project Structure

```
root_dir/
├── cifar_mps/   # Main package
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

### Training
```bash
uv run python -m scripts.train --model resnet18 
```
## Project Components

- **`cifar_mps/`** - Core implementation of all cifar related things
- **`scripts/`** - Training scripts and experiment utilities  
