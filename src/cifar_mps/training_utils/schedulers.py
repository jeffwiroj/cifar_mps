import math


class CosineWarmupScheduler:
    def __init__(self, optimizer, max_lr, max_steps, warmup_ratio=0.03, min_lr=1e-6):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.max_steps = max_steps
        self.warmup_steps = int(max_steps * warmup_ratio)
        self.min_lr = min_lr
        self.current_step = 0

        # Set initial learning rate to min_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr

    def get_lr(self, iteration=None):
        if iteration is None:
            iteration = self.current_step

        if iteration < self.warmup_steps:
            # Linear warmup
            return self.max_lr * (iteration / self.warmup_steps)
        else:
            # Cosine annealing
            cosine_iter = iteration - self.warmup_steps
            cosine_total = self.max_steps - self.warmup_steps
            return self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (
                1 + math.cos(math.pi * cosine_iter / cosine_total)
            )

    def step(self):
        lr = self.get_lr(self.current_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.current_step += 1
        return lr

    def get_last_lr(self):
        return [self.get_lr(self.current_step - 1)]
