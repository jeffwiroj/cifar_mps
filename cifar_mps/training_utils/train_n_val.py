import torch
import torch.nn as nn
import wandb
from dataclasses import asdict
from cifar_mps.config import TrainConfig, ExpConfig, get_run_name
from cifar_mps.training_utils.train_n_val import AverageMeter

def evaluate(model, data_loader, criterion):
    """Evaluate model on a given dataset and return avg loss and accuracy."""

    device = next(iter(model.parameters())).device  # grab the device type
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    model.eval()
    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            correct = (preds == labels).sum().item()
            accuracy = correct / images.size(0)

            loss_meter.update(loss.item(), n=images.size(0))
            acc_meter.update(accuracy, n=images.size(0))

    return loss_meter.avg, acc_meter.avg


def train_n_val(
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    train_config: TrainConfig,
    exp_config: ExpConfig,
    device,
):

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler() if exp_config.mixed_precision else None

    run = None
    run_name = get_run_name(train_config,exp_config)
    if exp_config.use_wandb:
        run = wandb.init(
            project=exp_config.exp_name,
            config=asdict(train_config),
            run_name=run_name
        )

    total_steps = len(train_loader)*train_config.epochs
    global_step = 0
    eval_iterval =  len(train_loader) // 2
    
    train_loss_meter = AverageMeter("train_loss")
    train_acc_meter = AverageMeter("train_acc")

    for epoch in range(train_config.epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Forward + Backward + Optimize
            optimizer.zero_grad()

            # Mixed Precision training
            if scaler:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(x)
                    loss = criterion(outputs, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
            if scheduler:
                scheduler.step()

            # Calculate accuracy of batch:
            # After getting outputs and loss
            B = x.size(0)
            pred = outputs.argmax(dim=1)
            correct = (pred == y).sum().item()
            accuracy = correct / B * 100  # Convert to percentage

            
            train_loss_meter.update(loss.item()*B,n=B)
            train_acc_meter.update(accuracy * B, n=B)
            if global_step % eval_iterval == 0:
                val_loss, val_acc = evaluate(model, val_loader, criterion)
                if run:
                    run.log(
                        {
                            "train_loss": train_loss_meter.avg,
                            "train_acc": train_acc_meter.avg ,
                             "val_loss": val_loss,
                             "val_acc": val_acc
                        })
                      

class AverageMeter:
    """Computes and stores the average, current value, sum, and count, with a name."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        """Resets all the meter values to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the meter with a new value.

        Args:
            val (float): New value to add.
            n (int): Weight of the new value (e.g., batch size). Default is 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """String representation."""
        return f"{self.name}: val={self.val:.4f}, avg={self.avg:.4f}"

    def summary(self):
        """Prints a formatted summary."""
        print(f"[{self.name}] Final Average: {self.avg:.4f} over {self.count} samples")
