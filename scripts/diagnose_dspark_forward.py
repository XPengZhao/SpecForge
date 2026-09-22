"""Fixed-anchor forward comparison of trusted DSpark checkpoints; no training."""
import argparse
from collections import Counter
import gc
from itertools import islice
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.modeling.target.offline_config import load_offline_target_config
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.runtime.data_plane.deepspec_cache import DeepSpecCacheReader, read_deepspec_features
from diagnose_dspark_checkpoints import load


def magnitude(value):
    x = value.detach().float()
    if not torch.isfinite(x).all():
        raise ValueError('Nonfinite activation')
    return dict(rms=x.square().mean().sqrt().item(), abs_max=x.abs().max().item())


def token_metrics(logits, labels, teacher):
    logp = logits.float().log_softmax(-1)
    p = logp.exp()
    return dict(ce=F.nll_loss(logp, labels, reduction='none'),
        acc=logits.argmax(-1).eq(labels).float(),
        entropy=-(p*logp).sum(-1), overlap=torch.minimum(p, teacher).sum(-1),
        centered_logit_rms=logits.float().std(-1, unbiased=False))


def fixed_anchors(model, ids, mask, count, seed):
    candidates = model._build_anchor_candidate_mask(ids.shape[1], mask)[0].nonzero().flatten().cpu()
    g = torch.Generator().manual_seed(seed)
    selected = candidates[torch.randperm(len(candidates), generator=g)[:count]].sort().values
    return selected[None].to(ids.device)


@torch.inference_mode()
def evaluate(model, samples, args):
    totals, counts, predictions, activations, anchors_saved = {}, {}, {}, {}, []
    hooks = []
    current = {}
    def hook(name):
        def save(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            current[name] = magnitude(value)
        return save
    def pre_hook(name):
        def save(module, inputs):
            current[name] = magnitude(inputs[0])
        return save
    for name, module in model.draft_model.named_modules():
        if (name in ('fc', 'hidden_norm', 'norm') or
            name.endswith(('q_proj', 'k_proj', 'v_proj', 'o_proj', 'q_norm', 'k_norm',
                           'input_layernorm', 'post_attention_layernorm',
                           'gate_proj', 'up_proj', 'down_proj', 'self_attn', 'mlp')) or
            (name.startswith('layers.') and name.count('.') == 1)):
            hooks.append(module.register_forward_hook(hook(name)))
        if name.endswith(('input_layernorm', 'post_attention_layernorm', 'down_proj', 'o_proj')):
            hooks.append(module.register_forward_pre_hook(pre_hook(name + '.input')))
    try:
        for sample_id, row in enumerate(samples):
            current.clear()
            ids = row['input_ids'][None].to(args.device)
            mask = row['loss_mask'][None].to(args.device)
            anchors = fixed_anchors(model, ids, mask, args.anchors, args.seed+sample_id)
            anchors_saved.append(anchors.cpu().tolist()[0])
            if not anchors.numel():
                continue
            keep = torch.ones_like(anchors, dtype=torch.bool)
            _, _, hidden = model._forward_draft_blocks(
                input_ids=ids, hidden_states=row['aux_hidden_state'][None].to(args.device),
                loss_mask=mask, anchor_positions=anchors, block_keep_mask=keep,
                ngram_embedding=row['ngram_embedding'][None].to(args.device) if 'ngram_embedding' in row else None)
            labels, valid, indices = model._build_dspark_labels_and_mask(ids, mask, anchors, keep)
            prev = torch.cat([ids.gather(1, anchors)[..., None], labels[:, :, :-1]], -1).reshape(-1)
            labels, valid, indices = labels.reshape(-1), valid.reshape(-1), indices.reshape(-1)
            h = hidden.reshape(-1, hidden.shape[-1])
            positions = torch.arange(len(labels), device=args.device) % model.block_size
            for start in range(0, len(labels), args.chunk_size):
                end = min(start+args.chunk_size, len(labels)); select = valid[start:end]
                if not select.any():
                    continue
                hh = h[start:end]
                base = model.lm_head(hh)
                full = model.draft_model.apply_logits_head(base[None],
                    prev_token_ids=prev[start:end][None], hidden_states=hh[None])[0]
                teacher_h = row['hidden_state'][(indices[start:end]-1).cpu()].to(args.device)
                teacher = model.lm_head(teacher_h).float().softmax(-1)
                bias = full.float()-base.float()
                for branch, logits in [('backbone', base), ('final', full)]:
                    metrics = token_metrics(logits, labels[start:end], teacher)
                    for scope in ['all'] + [f'mtp_{i+1}' for i in range(model.block_size)]:
                        chosen = select if scope == 'all' else select & (positions[start:end] == int(scope[4:])-1)
                        key = f'{branch}/{scope}'
                        counts[key] = counts.get(key, 0) + int(chosen.sum())
                        accum = totals.setdefault(key, {})
                        for metric, v in metrics.items():
                            accum[metric] = accum.get(metric, 0.) + v[chosen].sum().item()
                    counter = predictions.setdefault(branch, Counter())
                    counter.update(logits.argmax(-1)[select].cpu().tolist())
                key='bias/all'; counts[key]=counts.get(key,0)+int(select.sum())
                accum=totals.setdefault(key,{})
                for name, v in dict(centered_logit_rms=bias.std(-1,unbiased=False),
                                    abs_max=bias.abs().amax(-1)).items():
                    accum[name]=accum.get(name,0.)+v[select].sum().item()
            activations[str(sample_id)] = dict(current)
            print(f'  sample {sample_id+1}/{len(samples)}', flush=True)
    finally:
        for handle in hooks:
            handle.remove()
    return dict(metrics={k: dict(tokens=counts[k], **{m:v/counts[k] for m,v in values.items()})
                         for k,values in totals.items() if counts[k]},
                top_prediction_ids={k:c.most_common(10) for k,c in predictions.items()},
                activations_by_sample=activations, anchors_by_sample=anchors_saved)


def run(args):
    config=Qwen3Config.from_json_file(args.draft_config)
    config._attn_implementation=args.attention_backend
    reader=DeepSpecCacheReader(args.cache)
    if reader.layers != config.dflash_config['target_layer_ids'] or reader.hidden_size != config.hidden_size:
        raise ValueError('Draft aux config does not match cache')
    if config.dflash_config.get('ngram_mask', False): reader.enable_ngram()
    refs=list(islice(iter(reader),args.samples))
    if not refs: raise ValueError('No cache samples')
    samples=[read_deepspec_features(ref, reader.feature_keys) for ref in refs]
    target=TargetEmbeddingsAndHead.from_pretrained(args.target_model_path,
        config=load_offline_target_config(args.target_model_path),
        embed_key='model.language_model.embed_tokens.weight',lm_head_key='lm_head.weight',
        device=args.device,dtype=torch.bfloat16)
    draft=DSparkDraftModel(config).to(device=args.device,dtype=torch.bfloat16)
    model=OnlineDSparkModel(draft,target.lm_head,target.embed_tokens,
        mask_token_id=config.dflash_config['mask_token_id'],block_size=config.block_size,
        attention_backend=args.attention_backend,num_anchors=args.anchors).eval()
    model.requires_grad_(False)
    root=Path(args.run_dir)
    report=dict(draft_config=json.loads(Path(args.draft_config).read_text()),
        arguments=vars(args),sample_ids=[r.sample_id for r in refs],checkpoints={},
        note='Fixed-anchor teacher-forced diagnostic; averages are unweighted valid tokens, not training-window metrics or online MAL. Bias measured after actual BF16 addition. Snapshot changes do not establish the triggering cause.')
    for step in args.steps:
        paths=[p for p in root.glob(f'*-step{step}') if p.is_dir() and not p.is_symlink()]
        if len(paths)!=1: raise ValueError(f'Expected one checkpoint for {step}, got {paths}')
        print(f'Loading step {step}',flush=True)
        state=load(paths[0]/'training_state.pt')
        if state['global_step']!=step: raise ValueError('Checkpoint step mismatch')
        draft.load_state_dict(state['draft_state_dict'],strict=True)
        del state; gc.collect()
        result=evaluate(model,samples,args)
        report['checkpoints'][str(step)]=result
        print(json.dumps({k:v for k,v in result['metrics'].items() if k.endswith('/all')},indent=2),flush=True)
        out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'Report: {args.output}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True); p.add_argument('--cache',required=True)
    p.add_argument('--target-model-path',required=True); p.add_argument('--draft-config',required=True)
    p.add_argument('--steps',nargs='+',type=int,default=[651,1302,2604])
    p.add_argument('--samples',type=int,default=16); p.add_argument('--anchors',type=int,default=32)
    p.add_argument('--chunk-size',type=int,default=32); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--attention-backend',choices=['flex_attention','sdpa'],default='flex_attention')
    p.add_argument('--output',default='outputs/dspark-forward-diagnosis.json')
    args=p.parse_args()
    if min(args.samples,args.anchors,args.chunk_size)<1: p.error('Counts must be positive')
    run(args)
