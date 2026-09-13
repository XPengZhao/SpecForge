"""Bounded, detached DSpark logging-window statistics (not training losses)."""

import torch
import torch.distributed as dist


class MetricWindow:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sums = {}
        self.denoms = {}
        self.weights = {}
        self.local_loss_sum = None
        self.count = 0

    def update(self, payload):
        if payload is None:
            return
        sums, denoms, weights = (payload[k] for k in ("sums", "denoms", "weights"))
        if self.count and (set(sums) != set(self.sums) or weights != self.weights):
            raise ValueError("Logging metric schema changed inside a window")
        self.weights = dict(weights)
        for name in sums:
            num, den = sums[name].detach().float(), denoms[name].detach().float()
            if self.count:
                self.sums[name].add_(num)
                self.denoms[name].add_(den)
            else:
                self.sums[name], self.denoms[name] = num.clone(), den.clone()
        # Match DeepSpec's local component means, then average over emits.
        local_loss = sum(
            (weight * sums[name].detach().float() / (denoms[name].detach().float() + 1e-6)
             for name, weight in weights.items()),
            next(iter(sums.values())).detach().float().new_zeros(()),
        )
        if self.local_loss_sum is None:
            self.local_loss_sum = local_loss.clone()
        else:
            self.local_loss_sum.add_(local_loss)
        self.count += 1

    def summary(self, *, reset=True):
        if not self.count:
            return None
        names = sorted(self.sums)
        stats = torch.stack([self.sums[n] for n in names] + [self.denoms[n] for n in names])
        local_mean = self.local_loss_sum / self.count
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats)
            # DeepSpec prints rank 0's local-window mean; broadcast so every
            # logger sees the same summary while component ratios use all ranks.
            dist.broadcast(local_mean, src=0)
        size = len(names)
        values = stats[:size] / stats[size:].clamp_min(1e-6)
        result = dict(zip(names, values.cpu().tolist()))
        result['loss'] = local_mean.item()
        result['loss_weighted'] = sum(w * result[n] for n, w in self.weights.items())
        if 'tau_loss' in result:
            result['tau_loss_weighted'] = self.weights.get('tau_loss', 0.0) * result['tau_loss']
        result['log_micro_batches'] = self.count  # per rank, not global samples
        if 'acc' in result:
            result['accuracy_denom'] = stats[size + names.index('acc')].item()
        for i in range(2, 1000):
            current, previous = f'mtp_{i}_loss', f'mtp_{i-1}_loss'
            if current not in result or previous not in result:
                break
            result[f'mtp_{i}_minus_{i-1}_loss'] = result[current] - result[previous]
        if reset:
            self.reset()
        return result
