"""Read-only CPU audit of trusted local SpecForge checkpoints; no model construction."""
import argparse
import gc
import json
import math
import re
from pathlib import Path

import torch


def stats(t):
    flat = t.detach().reshape(-1)
    n, bad, negative, total, square, peak = flat.numel(), 0, 0, 0., 0., 0.
    for part in flat.split(1_000_000):
        x = part.double()
        finite = torch.isfinite(x)
        bad += int((~finite).sum())
        x = x[finite]
        negative += int((x < 0).sum())
        total += x.sum().item()
        square += x.square().sum().item()
        if x.numel():
            peak = max(peak, x.abs().max().item())
    result = dict(shape=list(t.shape), dtype=str(t.dtype), numel=n, nonfinite=bad,
                  negative=negative, mean=total/max(n-bad, 1),
                  rms=math.sqrt(square/max(n-bad, 1)), abs_max=peak)
    if n <= 32:
        result['values'] = [float(v) if math.isfinite(float(v)) else str(float(v)) for v in flat]
    return result


def difference(a, b):
    if a.shape != b.shape:
        return {'shape_mismatch': True}
    e, aa, bb, dot, bad = 0., 0., 0., 0., 0
    for x, y in zip(a.reshape(-1).split(1_000_000), b.reshape(-1).split(1_000_000)):
        x, y = x.double(), y.double()
        valid = torch.isfinite(x) & torch.isfinite(y)
        bad += int((~valid).sum())
        x, y = x[valid], y[valid]
        e += (y-x).square().sum().item()
        aa += x.square().sum().item(); bb += y.square().sum().item()
        dot += (x*y).sum().item()
    return dict(nonfinite_pairs=bad, delta_l2=math.sqrt(e),
                relative_delta=math.sqrt(e/aa) if aa else None,
                rms_ratio=math.sqrt(bb/aa) if aa else None,
                cosine=dot/math.sqrt(aa*bb) if aa and bb else None)


def load(path):
    # Checkpoints include Python/RNG metadata. Use only your own trusted files.
    try:
        return torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    except RuntimeError as error:
        if 'mmap can only be used with files saved with' not in str(error):
            raise
        print(f'Legacy serialization: loading {path} into CPU RAM (no mmap).', flush=True)
        return torch.load(path, map_location='cpu', weights_only=False)


def optimizer_for(path, shared):
    opt = shared.get('replicated_optimizer_state')
    if opt is not None:
        return opt, 'shared replicated state'
    rank = path / 'training_state_rank0.pt'
    if rank.exists():
        return load(rank).get('optimizer'), 'rank0 only (may be a shard)'
    return None, 'missing'


def optimizer_summary(opt):
    if opt is None:
        return None
    raw = opt.get('optimizer_state_dict', {})
    groups = [{k: v for k, v in g.items() if k != 'params'} for g in raw.get('param_groups', [])]
    states = {}
    for pid, state in raw.get('state', {}).items():
        states[str(pid)] = {k: stats(v) if torch.is_tensor(v) else v for k, v in state.items()}
    masters = {str(i): stats(v) for i, v in enumerate(opt.get('fp32_params', []))}
    return dict(param_groups=groups, max_grad_norm=opt.get('max_grad_norm'),
                states_by_optimizer_id=states, fp32_masters_by_index=masters)


def run(args):
    root = Path(args.run_dir)
    paths = {}
    for p in root.iterdir():
        match = re.search(r'-step(\d+)$', p.name)
        if match and p.is_dir() and not p.is_symlink():
            paths[int(match[1])] = p
    print('Available steps:', sorted(paths), flush=True)
    selected = sorted(set(args.steps)) if args.steps else sorted(paths)
    missing = [s for s in selected if s not in paths]
    if missing:
        raise ValueError(f'Missing checkpoints {missing}; available: {sorted(paths)}')
    if not selected:
        raise ValueError('No checkpoint directories found')
    report = dict(run_dir=str(root), checkpoints=[], comparisons=[], notes=[
        'CPU mmap scan of saved snapshots. Does not measure the actual collapse-step update.',
        'Optimizer IDs are not assigned to module names: saved optimizer has no name mapping.',
        'Large changes are leads, not proof of failure; normal training also changes weights.',
        'Rank0 optimizer fallback is not a full audit of sharded optimizer states.'])
    previous, previous_step = None, None
    for step in selected:
        path = paths[step]
        print(f'Reading step {step}: {path}', flush=True)
        state = load(path / 'training_state.pt')
        if state.get('global_step') != step:
            raise ValueError(f'Step metadata mismatch in {path}')
        weights = state['draft_state_dict']
        opt, origin = optimizer_for(path, state)
        summary = dict(step=step, optimizer_source=origin,
            weights={k: stats(v) for k, v in weights.items() if torch.is_tensor(v)},
            optimizer=optimizer_summary(opt))
        report['checkpoints'].append(summary)
        if previous is not None:
            changes = {k: difference(previous[k], v) for k, v in weights.items()
                       if k in previous and torch.is_tensor(v) and torch.is_tensor(previous[k])}
            ranked = sorted(changes, key=lambda k: changes[k].get('relative_delta') or 0., reverse=True)
            report['comparisons'].append(dict(from_step=previous_step, to_step=step,
                added_keys=sorted(weights.keys()-previous.keys()),
                removed_keys=sorted(previous.keys()-weights.keys()), weights=changes,
                largest_relative_changes=ranked[:20]))
            print(f'Largest relative changes {previous_step} -> {step}:', flush=True)
            for k in ranked[:10]:
                print(k, changes[k], flush=True)
        bad = [k for k, s in summary['weights'].items() if s['nonfinite']]
        print('Nonfinite weight tensors:', bad, flush=True)
        if summary['optimizer']:
            issues = [(pid, name) for pid, entry in summary['optimizer']['states_by_optimizer_id'].items()
                      for name, v in entry.items() if isinstance(v, dict) and
                      (v.get('nonfinite', 0) or (name in ('exp_avg_sq', 'max_exp_avg_sq') and v.get('negative', 0)))]
            print('Optimizer nonfinite/negative-second-moment entries:', issues, flush=True)
        for k, s in summary['weights'].items():
            if 'ngram' in k and s['numel'] <= 32:
                print(k, s, flush=True)
        previous, previous_step = weights, step
        del state, opt
        gc.collect()
    dest = Path(args.output) if args.output else root / 'checkpoint_diagnosis.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Report: {dest}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--steps', nargs='+', type=int)
    p.add_argument('--output')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.threads < 1:
        p.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    run(args)
