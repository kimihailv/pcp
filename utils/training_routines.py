import torch
from math import cos, pi
from collections import defaultdict


def get_warmup_schedule(
        optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + cos(pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class RunningMetrics:
    def __init__(self):
        self.current_step = 0
        self.metrics = defaultdict(lambda: 0)

    def step(self, metrics):
        for metric_name, val in metrics.items():
            self.metrics[metric_name] += val
        self.current_step += 1

    def report(self, prefix=''):
        report = {}
        if prefix != '':
            prefix += '_'
        for metric_name, val in self.metrics.items():
            report[f'{prefix}{metric_name}'] = val.item() / self.current_step

        return report
