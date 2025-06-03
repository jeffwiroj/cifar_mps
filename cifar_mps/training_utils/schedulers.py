import math


""" All schedulers here updates at a step level only
"""

class CosineWarmupScheduler:
    def __init__(self, optimizer, max_lr, max_steps, warmup_ratio=0.03, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.max_steps = max_steps
        self.warmup_steps = int(max_steps * warmup_ratio)
        self.min_lr = max_lr * min_lr_ratio
        self.current_step = 0
        
        # Set initial learning rate to very small value for warmup
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.max_lr / self.warmup_steps if self.warmup_steps > 0 else self.max_lr
    
    def get_lr(self, step=None):
        if step is None:
            step = self.current_step
            
        # 1) Linear warmup
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps
        
        # 2) Past max steps, return min learning rate
        if step > self.max_steps:
            return self.min_lr
        
        # 3) Cosine decay
        decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)
    
    def step(self):
        lr = self.get_lr(self.current_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_step += 1
        return lr
    
    def get_last_lr(self):
        return [self.get_lr(self.current_step - 1)]