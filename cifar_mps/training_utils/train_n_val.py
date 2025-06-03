import torch
import torch.nn as nn


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


def train_n_val(model, optimizer, scheduler, train_loader, val_loader, config,device):

    total_iters = len(train_loader) * config.epochs
    eval_every_n_iters = int(total_iters * 0.1)
    current_iter = 0

    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        train_loss, train_acc = evaluate(model, train_loader, criterion)
        test_loss, test_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch: {epoch + 1}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val  Loss: {test_loss:.4f} | Val  Acc: {test_acc:.4f}")


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
