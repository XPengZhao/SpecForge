"""Greedy synchronous reference for an exported dense 4+1 DSpark checkpoint.

Run from the repository root: PYTHONPATH=. python scripts/generate_dspark_4plus1.py ...
This checks model correctness/acceptance, not overlapped serving throughput.
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from specforge.modeling.draft.dspark import DSparkDraftModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', required=True)
    parser.add_argument('--draft', required=True, help='HF export directory')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--enable-thinking', action='store_true')
    args = parser.parse_args()
    dtype = torch.bfloat16 if args.device.startswith('cuda') else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=dtype, attn_implementation='sdpa').to(args.device).eval()
    draft = DSparkDraftModel.from_pretrained(
        args.draft, torch_dtype=dtype, attn_implementation='sdpa').to(args.device).eval()
    if not draft.late_conditioning:
        parser.error('--draft must be a 4+1 checkpoint')
    prompt = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': args.prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=args.enable_thinking)
    ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids.to(args.device)
    eos = target.generation_config.eos_token_id
    stops = eos if isinstance(eos, list) else ([] if eos is None else [eos])
    result = draft.spec_generate(target, ids, args.max_new_tokens, stops, 0., return_stats=True)
    output = result.pop('output_ids')
    result['text'] = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
    result['generated_tokens'] = output.shape[1] - ids.shape[1]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
