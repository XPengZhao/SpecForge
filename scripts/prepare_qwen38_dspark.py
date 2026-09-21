"""Generate offline DSpark configs from the actual Qwen3.8 checkpoint/cache."""
import argparse
import json
from pathlib import Path

import yaml
from safetensors import safe_open

from specforge.config.schema import Config
from specforge.modeling.target.offline_config import load_offline_target_config
from specforge.runtime.data_plane.deepspec_cache import DeepSpecCacheReader

ROOT = Path(__file__).resolve().parents[1]


def prepare(target, cache, output, mask_token_id=None, world_size=4, global_batch=512):
    target, cache, output = Path(target).resolve(), Path(cache).resolve(), Path(output).resolve()
    config = load_offline_target_config(str(target))
    reader = DeepSpecCacheReader(str(cache), max_len=4096)
    reader.validate_model(hidden_size=2560, target_layer_ids=[45, 46, 47],
                          target_model_path=str(target), target_config=config)
    if config.model_type != 'qwen4_exp':
        raise ValueError('Expected a Qwen3.8 qwen4_exp checkpoint')
    text = config.text_config
    embed_key, head_key = 'model.language_model.embed_tokens.weight', 'lm_head.weight'
    index = target / 'model.safetensors.index.json'
    weights = json.loads(index.read_text())['weight_map'] if index.exists() else {}
    for key in ([embed_key] if config.tie_word_embeddings else [embed_key, head_key]):
        shard = target / weights[key] if index.exists() else target / 'model.safetensors'
        with safe_open(str(shard), framework='pt', device='cpu') as f:
            if f.get_slice(key).get_shape() != [text.vocab_size, text.hidden_size]:
                raise ValueError(f'Unexpected checkpoint shape for {key}')
    if mask_token_id is None:
        tokenizer_path = target / 'tokenizer_config.json'
        tokens = json.loads(tokenizer_path.read_text()).get('added_tokens_decoder', {}) if tokenizer_path.exists() else {}
        for token in ('<|fim_pad|>', '<|mask|>'):
            matches = [int(k) for k, v in tokens.items() if v.get('content') == token]
            if matches:
                mask_token_id = matches[0]
                break
    if mask_token_id is None:
        raise ValueError('No recognized MASK token; supply --mask-token-id using an existing vocabulary ID')
    if not 0 <= mask_token_id < text.vocab_size:
        raise ValueError('mask-token-id must be inside the target vocabulary')
    if world_size <= 0 or global_batch < world_size or global_batch % world_size:
        raise ValueError('global-batch must be a positive multiple of world-size (micro batch 1)')
    steps = reader.num_samples // global_batch
    if steps < 1:
        raise ValueError('Cache has fewer samples than one global batch')
    draft = json.loads((ROOT / 'configs/qwen3.8-flash-next-dspark.json').read_text())
    draft.update(vocab_size=text.vocab_size, num_target_layers=48)
    # Match the target's full-attention head geometry, not its hybrid backbone.
    for key in ('num_attention_heads', 'num_key_value_heads', 'head_dim'):
        value = getattr(text, key, None)
        if type(value) is not int or value <= 0:
            raise ValueError(f'Missing or invalid target text_config.{key}')
        draft[key] = value
    if draft['num_attention_heads'] % draft['num_key_value_heads']:
        raise ValueError('Q head count must be divisible by KV head count')
    for key in ('bos_token_id', 'eos_token_id', 'pad_token_id'):
        draft[key] = getattr(text, key, None)
    draft['dflash_config'].update(mask_token_id=mask_token_id, target_layer_ids=[45, 46, 47])
    train = yaml.safe_load((ROOT / 'examples/configs/qwen3-4b-dspark-deepspec-offline.yaml').read_text())
    train['model'].update(target_model_path=str(target), embedding_key=embed_key,
                          lm_head_key=head_key, draft_model_config=str(output / 'draft_config.json'))
    train['data'].update(hidden_states_path=str(cache), dspark_supervision='response')
    train['training'].update(accumulation_steps=global_batch // world_size,
                             total_steps=steps * 10, max_steps=steps, save_interval=steps)
    train['deployment']['trainer']['nproc_per_node'] = world_size
    train.update(run_id='qwen3.8-flash-next-dspark-aux3',
                 output_dir='outputs/qwen3.8-flash-next-dspark-aux3', tracking={'report_to': 'tensorboard'})
    Config.model_validate(train)
    paths = [output / 'draft_config.json', output / 'train.yaml']
    if any(p.exists() for p in paths):
        raise FileExistsError('Generated configs already exist; choose a new --output-dir')
    output.mkdir(parents=True, exist_ok=True)
    paths[0].write_text(json.dumps(draft, indent=2) + '\n')
    paths[1].write_text(yaml.safe_dump(train, sort_keys=False))
    print(f'Prepared {paths[1]}: samples={reader.num_samples}, global_batch={global_batch}, '
          f'max_steps={steps}, schedule_steps={steps * 10}, mask_token_id={mask_token_id}')
    return draft, train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-model-path', required=True)
    parser.add_argument('--hidden-states-path', required=True)
    parser.add_argument('--output-dir', default='outputs/qwen38-setup')
    parser.add_argument('--mask-token-id', type=int)
    parser.add_argument('--world-size', type=int, default=4)
    parser.add_argument('--global-batch', type=int, default=512)
    args = parser.parse_args()
    prepare(args.target_model_path, args.hidden_states_path, args.output_dir,
            args.mask_token_id, args.world_size, args.global_batch)
