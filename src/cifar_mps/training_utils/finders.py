import torch
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import defaultdict
import copy


class LRFinder:
    """
    Learning Rate Finder class to find optimal learning rate for training.

    Usage:
        lr_finder = LRFinder(model, optimizer, criterion)
        lr_finder.range_test(dataloader, start_lr=1e-7, end_lr=10, num_iter=100)
        lr_finder.plot()
        optimal_lr = lr_finder.suggest_lr()
    """

    def __init__(self, model, optimizer, criterion, device=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        # Handle device specification
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        print(f"LRFinder initialized with device: {self.device}")

        # Move model to device
        self.model.to(self.device)

        # Store original state (after moving to device)
        self.model_state = copy.deepcopy(model.state_dict())
        self.optimizer_state = copy.deepcopy(optimizer.state_dict())

        # Results storage
        self.history = defaultdict(list)
        self.best_loss = None

    def range_test(
        self,
        dataloader,
        start_lr=1e-7,
        end_lr=10,
        num_iter=100,
        smooth_f=0.05,
        diverge_th=5,
    ):
        """
        Perform learning rate range test.

        Args:
            dataloader: DataLoader for training data
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_iter: Number of iterations to run
            smooth_f: Smoothing factor for loss (0-1)
            diverge_th: Threshold for stopping if loss diverges
        """
        # Reset history
        self.history = defaultdict(list)
        self.best_loss = None

        # Ensure model is on correct device and in training mode
        self.model.to(self.device)
        self.model.train()

        # Calculate learning rate multiplier
        lr_mult = (end_lr / start_lr) ** (1 / (num_iter - 1))

        # Set initial learning rate
        self._set_lr(start_lr)

        # Create data iterator
        data_iter = iter(dataloader)

        for i in range(num_iter):
            # Get current learning rate
            current_lr = start_lr * (lr_mult**i)
            self._set_lr(current_lr)

            try:
                # Get next batch
                inputs, targets = next(data_iter)
            except StopIteration:
                # Restart iterator if we run out of data
                data_iter = iter(dataloader)
                inputs, targets = next(data_iter)

            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Smooth the loss
            if i == 0:
                self.best_loss = loss.item()
                smoothed_loss = loss.item()
            else:
                smoothed_loss = (
                    smooth_f * loss.item() + (1 - smooth_f) * self.history["loss"][-1]
                )
                if smoothed_loss < self.best_loss:
                    self.best_loss = smoothed_loss

            # Store results
            self.history["lr"].append(current_lr)
            self.history["loss"].append(smoothed_loss)
            self.history["raw_loss"].append(loss.item())

            # Check for divergence
            if smoothed_loss > diverge_th * self.best_loss:
                print(f"Stopping early at iter {i}, loss diverged")
                break

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Print progress
            if (i + 1) % max(1, num_iter // 10) == 0:
                print(
                    f"Iter {i+1}/{num_iter}, LR: {current_lr:.2e}, Loss: {smoothed_loss:.4f}"
                )

        print("Learning rate range test completed!")

    def plot(self, skip_start=10, skip_end=5, log_lr=True, show_lr=None):
        """
        Plot the learning rate finder results.

        Args:
            skip_start: Number of batches to skip at start
            skip_end: Number of batches to skip at end
            log_lr: Whether to use log scale for learning rate
            show_lr: Specific learning rate to highlight on plot
        """
        if len(self.history["lr"]) == 0:
            print("No data to plot. Run range_test first.")
            return

        # Prepare data
        lrs = (
            self.history["lr"][skip_start:-skip_end]
            if skip_end > 0
            else self.history["lr"][skip_start:]
        )
        losses = (
            self.history["loss"][skip_start:-skip_end]
            if skip_end > 0
            else self.history["loss"][skip_start:]
        )

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses, "b-", linewidth=2)

        if log_lr:
            plt.xscale("log")

        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder")
        plt.grid(True, alpha=0.3)

        # Highlight suggested learning rate
        if show_lr:
            plt.axvline(
                x=show_lr,
                color="red",
                linestyle="--",
                label=f"Suggested LR: {show_lr:.2e}",
            )
            plt.legend()

        plt.tight_layout()
        plt.show()

    def suggest_lr(self, skip_start=10, skip_end=5, method="valley"):
        """
        Suggest optimal learning rate based on the results.

        How to interpret the LR finder plot:
        1. Start: Loss stays flat (LR too small)
        2. Sweet spot: Loss decreases rapidly (good LR range)
        3. Minimum: Lowest loss point (often unstable)
        4. Divergence: Loss explodes (LR too high)

        Rule of thumb: Pick LR where loss is decreasing fastest,
        typically 1/10th to 1/3rd of the minimum loss LR.

        Args:
            skip_start: Number of batches to skip at start
            skip_end: Number of batches to skip at end
            method: Method to use ('valley', 'steep', 'minimum')

        Returns:
            Suggested learning rate
        """
        if len(self.history["lr"]) == 0:
            print("No data available. Run range_test first.")
            return None

        # Prepare data
        lrs = (
            np.array(self.history["lr"][skip_start:-skip_end])
            if skip_end > 0
            else np.array(self.history["lr"][skip_start:])
        )
        losses = (
            np.array(self.history["loss"][skip_start:-skip_end])
            if skip_end > 0
            else np.array(self.history["loss"][skip_start:])
        )

        if method == "minimum":
            # Find minimum loss
            min_idx = np.argmin(losses)
            suggested_lr = lrs[min_idx]

        elif method == "steep":
            # Find steepest gradient (derivative)
            gradients = np.gradient(losses)
            min_gradient_idx = np.argmin(gradients)
            suggested_lr = lrs[min_gradient_idx]

        elif method == "valley":
            # Find learning rate where loss is decreasing fastest
            # This is typically 1/10th of the learning rate at minimum loss
            min_idx = np.argmin(losses)
            suggested_lr = lrs[min_idx] / 10

        else:
            raise ValueError("Method must be one of: 'valley', 'steep', 'minimum'")

        print(f"Suggested learning rate ({method} method): {suggested_lr:.2e}")
        return suggested_lr

    def reset(self):
        """Reset model and optimizer to original state."""
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        print("Model and optimizer reset to original state")

    def _set_lr(self, lr):
        """Set learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


class BatchSizeFinder:
    """
    Batch Size Finder class to find optimal batch size for training.

    Usage:
        bs_finder = BatchSizeFinder(model, optimizer, criterion)
        bs_finder.range_test(dataset, lr=1e-3)
        optimal_bs = bs_finder.suggest_batch_size()
    """

    def __init__(self, model, optimizer, criterion, device=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        # Handle device specification
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        print(f"BatchSizeFinder initialized with device: {self.device}")

        # Move model to device
        self.model.to(self.device)

        # Store original state (after moving to device)
        self.model_state = copy.deepcopy(model.state_dict())
        self.optimizer_state = copy.deepcopy(optimizer.state_dict())

        # Results storage
        self.results = defaultdict(list)

    def range_test(
        self,
        dataset,
        lr=1e-3,
        start_bs=16,
        max_bs=1024,
        num_epochs=1,
        scaling_rule="linear",
    ):
        """
        Test different batch sizes to find optimal one.

        Args:
            dataset: Dataset to test with
            lr: Base learning rate (will be scaled with batch size)
            start_bs: Starting batch size
            max_bs: Maximum batch size to test
            num_epochs: Number of epochs to test each batch size
            scaling_rule: How to scale LR with batch size ('linear', 'sqrt', 'none')
        """
        # Reset results
        self.results = defaultdict(list)

        # Test powers of 2
        bs = start_bs
        base_lr = lr

        while bs <= max_bs:
            print(f"\nTesting batch size: {bs}")

            try:
                # Scale learning rate based on batch size
                if scaling_rule == "linear":
                    current_lr = base_lr * (bs / start_bs)
                elif scaling_rule == "sqrt":
                    current_lr = base_lr * np.sqrt(bs / start_bs)
                else:  # 'none'
                    current_lr = base_lr

                print(f"  Using LR: {current_lr:.2e}")

                # Create dataloader with current batch size
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=bs,
                    shuffle=True,
                    num_workers=min(4, bs // 8),
                    pin_memory=torch.cuda.is_available(),
                )

                # Reset model and optimizer
                self.model.load_state_dict(self.model_state)
                self.optimizer.load_state_dict(self.optimizer_state)
                self._set_lr(current_lr)

                # Time the training
                start_time = time.time()
                epoch_losses = []
                total_samples = 0

                # Clear GPU cache before testing
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                for epoch in range(num_epochs):
                    epoch_loss = 0
                    num_batches = 0

                    for inputs, targets in dataloader:
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)

                        self.optimizer.zero_grad()
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                        loss.backward()
                        self.optimizer.step()

                        epoch_loss += loss.item()
                        num_batches += 1
                        total_samples += len(inputs)

                    epoch_losses.append(epoch_loss / num_batches)

                end_time = time.time()
                total_time = end_time - start_time

                # Calculate metrics
                throughput = total_samples / total_time  # samples/second
                time_per_epoch = total_time / num_epochs
                final_loss = epoch_losses[-1]

                # Memory usage
                memory_gb = 0
                if torch.cuda.is_available():
                    memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

                # Store results
                self.results["batch_size"].append(bs)
                self.results["learning_rate"].append(current_lr)
                self.results["time_per_epoch"].append(time_per_epoch)
                self.results["throughput"].append(throughput)
                self.results["final_loss"].append(final_loss)
                self.results["memory_gb"].append(memory_gb)

                print(f"  Time per epoch: {time_per_epoch:.2f}s")
                print(f"  Throughput: {throughput:.1f} samples/sec")
                print(f"  Final loss: {final_loss:.4f}")
                print(f"  GPU memory: {memory_gb:.2f} GB")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  Out of memory at batch size {bs}")
                    break
                else:
                    raise e

            bs *= 2

        print("\nBatch size range test completed!")

    def plot(self):
        """Plot batch size test results."""
        if not self.results["batch_size"]:
            print("No data to plot. Run range_test first.")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        batch_sizes = self.results["batch_size"]

        # Time per epoch
        ax1.plot(
            batch_sizes, self.results["time_per_epoch"], "bo-", linewidth=2, markersize=8
        )
        ax1.set_xlabel("Batch Size")
        ax1.set_ylabel("Time per Epoch (s)")
        ax1.set_title("Training Speed vs Batch Size")
        ax1.set_xscale("log", base=2)
        ax1.grid(True, alpha=0.3)

        # Throughput
        ax2.plot(
            batch_sizes, self.results["throughput"], "go-", linewidth=2, markersize=8
        )
        ax2.set_xlabel("Batch Size")
        ax2.set_ylabel("Samples/Second")
        ax2.set_title("Throughput vs Batch Size")
        ax2.set_xscale("log", base=2)
        ax2.grid(True, alpha=0.3)

        # Final loss
        ax3.plot(
            batch_sizes, self.results["final_loss"], "ro-", linewidth=2, markersize=8
        )
        ax3.set_xlabel("Batch Size")
        ax3.set_ylabel("Final Loss")
        ax3.set_title("Loss vs Batch Size")
        ax3.set_xscale("log", base=2)
        ax3.grid(True, alpha=0.3)

        # Memory usage
        if any(m > 0 for m in self.results["memory_gb"]):
            ax4.plot(
                batch_sizes, self.results["memory_gb"], "mo-", linewidth=2, markersize=8
            )
            ax4.set_xlabel("Batch Size")
            ax4.set_ylabel("GPU Memory (GB)")
            ax4.set_title("Memory Usage vs Batch Size")
            ax4.set_xscale("log", base=2)
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(
                0.5,
                0.5,
                "GPU Memory\nNot Available",
                ha="center",
                va="center",
                transform=ax4.transAxes,
                fontsize=12,
            )
            ax4.set_title("Memory Usage")

        plt.tight_layout()
        plt.show()

    def suggest_batch_size(self, priority="throughput"):
        """
        Suggest optimal batch size based on specified priority.

        Args:
            priority: Optimization priority ('throughput', 'memory', 'loss', 'balanced')

        Returns:
            Suggested batch size
        """
        if not self.results["batch_size"]:
            print("No data available. Run range_test first.")
            return None

        batch_sizes = np.array(self.results["batch_size"])
        throughput = np.array(self.results["throughput"])
        memory = np.array(self.results["memory_gb"])
        losses = np.array(self.results["final_loss"])

        if priority == "throughput":
            best_idx = np.argmax(throughput)
            reason = f"highest throughput ({throughput[best_idx]:.1f} samples/sec)"

        elif priority == "memory":
            # Find largest batch size that uses reasonable memory
            memory_threshold = np.max(memory) * 0.8  # Use 80% of max available
            valid_indices = np.where(memory <= memory_threshold)[0]
            if len(valid_indices) > 0:
                best_idx = valid_indices[np.argmax(batch_sizes[valid_indices])]
                reason = f"memory efficient ({memory[best_idx]:.2f} GB)"
            else:
                best_idx = 0
                reason = "smallest memory footprint"

        elif priority == "loss":
            best_idx = np.argmin(losses)
            reason = f"lowest loss ({losses[best_idx]:.4f})"

        elif priority == "balanced":
            # Normalize metrics and find best balance
            norm_throughput = throughput / np.max(throughput)
            norm_loss = 1 - (losses / np.max(losses))  # Invert so higher is better
            norm_memory = 1 - (
                memory / np.max(memory)
            )  # Invert so lower memory is better

            # Weighted score (adjust weights as needed)
            score = 0.4 * norm_throughput + 0.4 * norm_loss + 0.2 * norm_memory
            best_idx = np.argmax(score)
            reason = f"balanced performance (score: {score[best_idx]:.3f})"

        else:
            raise ValueError(
                "Priority must be one of: 'throughput', 'memory', 'loss', 'balanced'"
            )

        suggested_bs = batch_sizes[best_idx]
        suggested_lr = self.results["learning_rate"][best_idx]

        print(f"\nBatch Size Recommendation ({priority} priority):")
        print(f"Batch size: {suggested_bs} - {reason}")
        print(f"Corresponding LR: {suggested_lr:.2e}")
        print(f"Expected throughput: {throughput[best_idx]:.1f} samples/sec")
        print(f"Expected memory: {memory[best_idx]:.2f} GB")
        print(f"Expected loss: {losses[best_idx]:.4f}")

        return suggested_bs, suggested_lr

    def reset(self):
        """Reset model and optimizer to original state."""
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        print("Model and optimizer reset to original state")

    def _set_lr(self, lr):
        """Set learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
