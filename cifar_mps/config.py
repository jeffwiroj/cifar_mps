import argparse
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TrainConfig:
    """Training configuration with organized sections"""

    # Model Section
    model_name: str = "smol_resnet"  # Name of the model architecture to use

    # Training Hyperparameters
    batch_size: int = 512
    epochs: int = 120
    max_lr: float = 0.04
    wd: float = 1e-4
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Optimizer Configuration
    optimizer: Literal["sgd", "adamw"] = "sgd"  # Optimizer type

    # Learning Rate Scheduling
    scheduler: Literal["none", "cosine", "linear"] = "none"  # LR scheduler type
    lr_warmup: float = (
        0.00  # Linear LR warmup as percentage of total iterations (0.0-1.0)
    )

    # Data Augmentation
    augmentations: str = "random"  # Type of data augmentations to apply


@dataclass
class ExpConfig:
    """Experiment configuration for logging and checkpointing"""

    exp_name: str = ""  # Name of the experiment for identification
    weight_folder: str = ""  # Directory to save model weights/checkpoints
    use_wandb: bool = False  # Whether to use Weights & Biases for logging
    verbose: bool = False  # If true, will log/prints more
    mixed_precision: bool = False  # Whether to use mixed precision (FP16) training


def parse_args() -> tuple[TrainConfig, ExpConfig]:
    """
    Parse command line arguments and return TrainConfig and ExpConfig objects.

    Returns:
        tuple: (TrainConfig, ExpConfig) objects populated with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Training configuration parser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model Section
    model_group = parser.add_argument_group("Model Section")
    model_group.add_argument(
        "--model-name",
        type=str,
        default="smol_resnet",
        help="Name of the model architecture to use",
    )

    # Training Hyperparameters
    train_group = parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument(
        "--batch-size", "-b", type=int, default=512, help="Batch size for training"
    )
    train_group.add_argument(
        "--epochs", "-e", type=int, default=40, help="Number of training epochs"
    )
    train_group.add_argument(
        "--max-lr", "-lr", type=float, default=0.1, help="Maximum learning rate"
    )
    train_group.add_argument("--wd", "-w", type=float, default=1e-4, help="Weight decay")
    train_group.add_argument(
        "--dropout", "-d", type=float, default=0.1, help="Dropout rate"
    )
    train_group.add_argument(
        "--grad-clip", "-g", type=float, default=1.0, help="Grad Norm clip"
    )

    # Optimizer Configuration
    opt_group = parser.add_argument_group("Optimizer Configuration")
    opt_group.add_argument(
        "--optimizer",
        "-o",
        type=str,
        choices=["sgd", "adamw"],
        default="sgd",
        help="Optimizer type",
    )

    # Learning Rate Scheduling
    sched_group = parser.add_argument_group("Learning Rate Scheduling")
    sched_group.add_argument(
        "--scheduler",
        "-s",
        type=str,
        choices=["none", "cosine", "linear"],
        default="none",
        help="Learning rate scheduler type",
    )
    sched_group.add_argument(
        "--lr-warmup",
        type=float,
        default=0.0,
        help="Linear LR warmup as percentage of total iterations (0.0-1.0)",
    )

    # Data Augmentation
    aug_group = parser.add_argument_group("Data Augmentation")
    aug_group.add_argument(
        "--augmentations",
        type=str,
        default="random",
        help="Type of data augmentations to apply",
    )

    # Experiment Configuration
    exp_group = parser.add_argument_group("Experiment Configuration")
    exp_group.add_argument(
        "--exp-name",
        "-n",
        type=str,
        default="supervised_cifar",
        help="Name of the experiment for identification",
    )
    exp_group.add_argument(
        "--weight-folder",
        type=str,
        default="",
        help="Directory to save model weights/checkpoints",
    )
    exp_group.add_argument(
        "--use-wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    exp_group.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    exp_group.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision (FP16) training",
    )
    args = parser.parse_args()

    # Create config objects
    train_config = TrainConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        lr_warmup=args.lr_warmup,
        epochs=args.epochs,
        augmentations=args.augmentations,
        max_lr=args.max_lr,
        wd=args.wd,
        dropout=args.dropout,
        grad_clip=args.grad_clip,
    )

    exp_config = ExpConfig(
        exp_name=args.exp_name,
        weight_folder=args.weight_folder,
        use_wandb=args.use_wandb,
        verbose=args.verbose,
        mixed_precision=args.mixed_precision,
    )
    return args, train_config, exp_config


def get_run_name(train_config: TrainConfig, exp_config: ExpConfig) -> str:
    """
    Generate a descriptive run name based on the most important configuration parameters.

    Args:
        train_config: Training configuration object
        exp_config: Experiment configuration object

    Returns:
        str: Formatted run name for experiment tracking
    """

    base_name = train_config.model_name

    # Format learning rate for readability (remove scientific notation for small values)
    lr_str = f"{train_config.max_lr:.4f}".rstrip("0").rstrip(".")

    # Format weight decay
    if train_config.wd >= 1e-3:
        wd_str = f"{train_config.wd:.4f}".rstrip("0").rstrip(".")
    else:
        wd_str = f"{train_config.wd:.0e}".replace("e-0", "e-").replace("e-", "e")

    # Build run name with most important hyperparameters
    run_name_parts = [
        base_name,
        f"bs{train_config.batch_size}",
        f"lr{lr_str}",
        f"wd{wd_str}",
        f"{train_config.optimizer}",
        f"e{train_config.epochs}",
    ]

    # Add scheduler if not 'none'
    if train_config.scheduler != "none":
        run_name_parts.append(f"sched_{train_config.scheduler}")

    # Add warmup if used
    if train_config.lr_warmup > 0:
        warmup_str = f"{train_config.lr_warmup:.2f}".rstrip("0").rstrip(".")
        run_name_parts.append(f"warmup{warmup_str}")

    # Add dropout if not default
    if train_config.dropout != 0.1:
        dropout_str = f"{train_config.dropout:.2f}".rstrip("0").rstrip(".")
        run_name_parts.append(f"drop{dropout_str}")
    return "_".join(run_name_parts)
