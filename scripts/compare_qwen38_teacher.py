"""Compare aligned DeepSpec cache prefixes using only a local LM head (no target forward)."""
import argparse
import json
from itertools import islice
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from specforge.runtime.data_plane.deepspec_cache import DeepSpecCacheReader, read_deepspec_features


def distribution_metrics(a, b):
    """FP32 log probabilities; KL direction is reference -> candidate."""
    la, lb = F.log_softmax(a.float(), -1), F.log_softmax(b.float(), -1)
    p, q = la.exp(), lb.exp()
    return {
        'overlap': torch.minimum(p, q).sum(-1),
        'kl_hf_to_sglang': (p * (la - lb)).sum(-1),
        'top1_agreement': (a.argmax(-1) == b.argmax(-1)).float(),
    }


def load_head(root, key, device):
    root = Path(root)
    index = root / 'model.safetensors.index.json'
    if index.exists():
        mapping = json.loads(index.read_text())['weight_map']
        paths = [root / mapping[key]]
    else:
        paths = sorted(root.glob('*.safetensors'))
    for path in paths:
        with safe_open(path, framework='pt', device='cpu') as f:
            if key in f.keys():
                return f.get_tensor(key).to(device=device, dtype=torch.float32)
    raise ValueError(f'Missing {key} in {root}')


def summarize(values):
    if not values:
        return {'tokens': 0}
    x = torch.cat(values).float()
    return dict(tokens=x.numel(), mean=x.mean().item(), min=x.min().item(),
                p01=x.quantile(.01).item(), p50=x.median().item(),
                p99=x.quantile(.99).item(), max=x.max().item())


@torch.inference_mode()
def run(args):
    readers = [DeepSpecCacheReader(p) for p in (args.reference, args.candidate)]
    for key in ('target_layer_ids', 'hidden_size', 'chat_template_sha256',
                'aux_reduction', 'target_final_feature'):
        if readers[0].manifest.get(key) != readers[1].manifest.get(key):
            raise ValueError(f'Metadata mismatch: {key}')
    n = min(args.max_samples, readers[0].num_samples)
    if readers[1].num_samples < n:
        raise ValueError('Candidate prefix is too short')
    head = load_head(args.target_model_path, args.head_key, args.device)
    if head.ndim != 2 or head.shape[1] != readers[0].hidden_size:
        raise ValueError('LM head width does not match cache')
    torch.backends.cuda.matmul.allow_tf32 = False
    stats = {'all_next_tokens': {}, 'response_next_tokens': {}}
    feature_stats, worst, samples = {}, [], []
    refs = [list(islice(iter(r), n)) for r in readers]
    keys = ['input_ids', 'loss_mask', 'aux_hidden_state', 'hidden_state']
    for sample, pair in enumerate(zip(*refs)):
        a, b = [read_deepspec_features(ref, keys) for ref in pair]
        for key in ('input_ids', 'loss_mask'):
            if not torch.equal(a[key], b[key]):
                raise ValueError(f'Sample {sample}: {key} mismatch')
        for key in ('aux_hidden_state', 'hidden_state'):
            if not torch.isfinite(a[key]).all() or not torch.isfinite(b[key]).all():
                raise ValueError(f'Sample {sample}: nonfinite {key}')
            slices = [('final', a[key], b[key])] if key == 'hidden_state' else [
                (f'aux_layer_{layer}', x, y) for layer, x, y in zip(
                    readers[0].layers, a[key].split(readers[0].hidden_size, -1),
                    b[key].split(readers[0].hidden_size, -1))]
            for name, x, y in slices:
                x, y = x.double(), y.double()
                s = feature_stats.setdefault(name, dict(error2=0., ref2=0., cand2=0., dot=0., max_abs=0.))
                s['error2'] += (x-y).square().sum().item()
                s['ref2'] += x.square().sum().item()
                s['cand2'] += y.square().sum().item()
                s['dot'] += (x*y).sum().item()
                s['max_abs'] = max(s['max_abs'], (x-y).abs().max().item())
        length = len(a['input_ids']) - 1
        sample_overlap = []
        for start in range(0, length, args.chunk_size):
            end = min(start + args.chunk_size, length)
            la, lb = [F.linear(row['hidden_state'][start:end].to(args.device, torch.float32), head)
                      for row in (a, b)]
            if not torch.isfinite(la).all() or not torch.isfinite(lb).all():
                raise ValueError('Nonfinite teacher logits')
            values = {k: v.cpu() for k, v in distribution_metrics(la, lb).items()}
            labels = a['input_ids'][start+1:end+1].to(args.device)
            for tag, logits in [('hf', la), ('sglang', lb)]:
                values[f'{tag}_next_token_nll'] = F.cross_entropy(logits, labels, reduction='none').cpu()
            mask = a['loss_mask'][start+1:end+1].bool()
            for scope, select in [('all_next_tokens', torch.ones_like(mask)), ('response_next_tokens', mask)]:
                for name, v in values.items():
                    stats[scope].setdefault(name, []).append(v[select])
            sample_overlap.append(values['overlap'])
            top_a, top_b = la.argmax(-1).cpu(), lb.argmax(-1).cpu()
            for j in torch.argsort(values['overlap'])[:args.worst_tokens].tolist():
                worst.append(dict(sample_index=sample, hidden_position=start+j,
                    predicted_position=start+j+1, response=bool(mask[j]),
                    target_token_id=int(labels[j]), hf_top1=int(top_a[j]), sglang_top1=int(top_b[j]),
                    overlap=float(values['overlap'][j]), kl=float(values['kl_hf_to_sglang'][j])))
            worst = sorted(worst, key=lambda r: r['overlap'])[:args.worst_tokens]
        samples.append(dict(sample_index=sample, length=length+1, overlap=summarize(sample_overlap)))
        print(f'Compared {sample+1}/{n}', flush=True)
    features = {k: dict(relative_l2=(s['error2']/max(s['ref2'], 1e-30))**.5,
        cosine=s['dot']/max((s['ref2']*s['cand2'])**.5, 1e-30), max_abs=s['max_abs'])
        for k, s in feature_stats.items()}
    result = dict(reference=args.reference, candidate=args.candidate,
        target_model_path=args.target_model_path, head_key=args.head_key,
        projection_dtype='float32; TF32 disabled', samples_compared=n,
        note='Matched cache prefix; stored trajectories, not online acceptance or MAL.',
        features=features, distributions={scope: {k: summarize(v) for k, v in metrics.items()}
        for scope, metrics in stats.items()}, samples=samples, worst_tokens=worst)
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['distributions'], indent=2))
    print(f'Report: {dest}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--target-model-path', required=True)
    p.add_argument('--head-key', default='lm_head.weight')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-samples', type=int, default=32)
    p.add_argument('--chunk-size', type=int, default=32)
    p.add_argument('--worst-tokens', type=int, default=30)
    p.add_argument('--output', default='teacher_comparison.json')
    args = p.parse_args()
    if min(args.max_samples, args.chunk_size, args.worst_tokens) < 1:
        p.error('Counts must be positive')
    run(args)
