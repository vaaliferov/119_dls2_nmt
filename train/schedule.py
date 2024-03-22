import math
import torch

def constant_learning_rate(lr, training_steps):
    return [lr] * training_steps

def linear_learning_rate(lr, warmup_steps, training_steps):
    w = lambda step: step / max(1, warmup_steps)
    c = lambda step: max(0.0, (training_steps - step) / max(1, training_steps - warmup_steps))
    return [lr * (c(step) if step > warmup_steps else w(step)) for step in range(training_steps)]

def cosine_learning_rate(lr, warmup_steps, training_steps, cycles=.5):
    w = lambda step: step / max(1, warmup_steps)
    p = lambda step: (step - warmup_steps) / max(1, training_steps - warmup_steps)
    c = lambda step: max(0.0, 0.5 * (1.0 + math.cos(math.pi * cycles * 2.0 * p(step))))
    return [lr * (c(step) if step > warmup_steps else w(step)) for step in range(training_steps)]

def noam_learning_rate(factor, hid_dim, warmup_steps, training_steps):
    r = lambda step: factor * hid_dim ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
    return [r(step) for step in range(1, training_steps + 1)]

class ScheduledAdam(torch.optim.Adam):
    def __init__(self, params, lr_schedule, **kwargs):
        super().__init__(params, **kwargs)
        self.param_groups[0]['lr'] = 0
        self.lr_schedule = lr_schedule
        self.cur_step = 0
    
    def step(self):
        lr = self.lr_schedule[self.cur_step]
        self.param_groups[0]['lr'] = lr
        self.cur_step += 1
        super().step()