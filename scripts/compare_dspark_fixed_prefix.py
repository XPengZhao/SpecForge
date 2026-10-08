"""Collect first-prefill vLLM traces, then replay identical DSpark blocks offline."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def prefix_digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()[:20]


def select_anchors(mask, k, count):
    valid = [a for a in range(1, len(mask) - k)
             if all(mask[a:a + k + 1])]
    if not valid:
        return []
    return sorted(set(valid[round(i * (len(valid) - 1) / max(count - 1, 1))]
                      for i in range(count)))


def matching_prefix(candidate, reference):
    if len(candidate) != len(reference):
        raise ValueError("Candidate/reference lengths differ")
    for i, (left, right) in enumerate(zip(candidate, reference)):
        if left != right:
            return i
    return len(candidate)


def selected_rows(args):
    rows = json.loads(Path(args.comparison).read_text())
    if not args.all_blocks:
        rows = [row for row in rows if not all(row["cache_aux_token_matches"])
                or not all(row["online_aux_token_matches"])]
    cases = {(case["cache_index"], case["anchor"]): case for case in
             (json.loads(line) for line in Path(args.cases).read_text().splitlines() if line.strip())}
    return [(row, cases[(row["cache_index"], row["anchor"])]) for row in rows]


def write_report(path, report):
    with Path(path).open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(path)


def inspect_scores(args):
    """Reconstruct baseline head scores from saved hidden, with equal previous tokens."""
    import torch
    from specforge.modeling.auto import AutoDraftModel
    from specforge.modeling.target.offline_config import load_offline_target_config
    from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    selected = selected_rows(args)
    draft, loading = AutoDraftModel.from_pretrained(args.checkpoint, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16, output_loading_info=True)
    if ([key for key in loading.get("missing_keys", []) if "embed_tokens" not in key]
            or loading.get("unexpected_keys") or loading.get("mismatched_keys")):
        raise ValueError(f"Incomplete weight loading: {loading}")
    if getattr(draft, "prefix_reranker", None) is not None:
        raise ValueError("inspect-scores currently supports vanilla baseline only")
    draft = draft.to(args.device).eval()
    target = TargetEmbeddingsAndHead.from_pretrained(args.target,
        config=load_offline_target_config(args.target), device=args.device, dtype=torch.bfloat16,
        embed_key="model.language_model.embed_tokens.weight", lm_head_key="lm_head.weight")
    reports = []
    with torch.inference_mode():
        for row, case in selected:
            trace = torch.load(case["trace"], map_location="cpu", weights_only=True)
            if Path(trace["draft_model"]).resolve() != Path(args.checkpoint).resolve():
                raise ValueError("Trace/checkpoint mismatch")
            positions = {}
            for name in ("cache_aux", "online_aux"):
                matches = row[name + "_token_matches"]
                if not all(matches):
                    positions.setdefault(matches.index(False), []).append(name)
            hidden = trace["draft_hidden"].to(args.device)
            entries = []
            for position, sources in positions.items():
                prev = trace["anchor_token_id"] if position == 0 else row["vllm_tokens"][position - 1]
                h = hidden[position:position + 1]
                scores = draft.markov_head.apply_step_logits(target.lm_head(h),
                    token_ids=torch.tensor([prev], device=args.device), hidden_states=h)[0].float()
                values, ids = scores.topk(5)
                chosen = row["vllm_tokens"][position]
                alternatives = {name: row["offline_" + name + "_tokens"][position] for name in sources}
                entries.append(dict(position_zero_based=position, previous_token=prev,
                    vllm_chosen_token=chosen, offline_alternatives=alternatives,
                    reconstructed_top5=[dict(token=int(t), score=float(v)) for t, v in zip(ids, values)],
                    reconstructed_top1_top2_gap=float(values[0] - values[1]),
                    reconstructed_argmax_matches_vllm=int(ids[0]) == chosen,
                    reconstructed_vllm_minus_alternative={name: float(scores[chosen] - scores[token])
                        for name, token in alternatives.items()}))
            reports.append(dict(cache_index=row["cache_index"], anchor=row["anchor"],
                score_source="offline target head + Markov on saved vLLM hidden; not original online logits",
                first_divergences=entries))
    write_report(args.output, reports)


def verify_target(args):
    """Greedy acceptance is the matching prefix against a target-only continuation."""
    import torch
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    reports = []
    for row, case in selected_rows(args):
        trace = torch.load(case["trace"], map_location="cpu", weights_only=True)
        k = len(row["vllm_tokens"])
        prefix = trace["prefix_token_ids"] + [trace["anchor_token_id"]]
        body = dict(model=args.model, prompt=prefix, max_tokens=k, temperature=0,
                    top_p=1., top_k=-1, repetition_penalty=1., presence_penalty=0.,
                    frequency_penalty=0., ignore_eos=True, return_token_ids=True)
        request = urllib.request.Request(args.url.rstrip("/") + "/v1/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.load(response)
        reference = result["choices"][0].get("token_ids")
        if not isinstance(reference, list) or len(reference) != k:
            raise ValueError("Target must return exactly k token_ids; check return_token_ids support")
        lengths = {name: matching_prefix(row[key], reference) for name, key in
                   (("vllm", "vllm_tokens"), ("cache_aux", "offline_cache_aux_tokens"),
                    ("online_aux", "offline_online_aux_tokens"))}
        entry = dict(cache_index=row["cache_index"], anchor=row["anchor"],
            target_greedy_tokens=reference, accepted_draft_tokens=lengths,
            acceptance_length_with_bonus={name: length + 1 for name, length in lengths.items()},
            cache_minus_vllm=lengths["cache_aux"] - lengths["vllm"],
            online_aux_minus_vllm=lengths["online_aux"] - lengths["vllm"])
        reports.append(entry)
        print(json.dumps(entry), flush=True)
    n = len(reports)
    summary = dict(blocks=n, scope="all blocks" if args.all_blocks else "only mismatching blocks",
        mean_cache_minus_vllm=sum(row["cache_minus_vllm"] for row in reports) / n if n else None,
        mean_online_aux_minus_vllm=sum(row["online_aux_minus_vllm"] for row in reports) / n if n else None)
    write_report(args.output, dict(summary=summary, blocks=reports))


def collect(args):
    import torch
    from specforge.runtime.data_plane.deepspec_cache import DeepSpecCacheReader, read_deepspec_features
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cases_path = output / "cases.jsonl"
    if cases_path.exists():
        raise FileExistsError(f"Choose a fresh output directory: {cases_path}")
    reader = DeepSpecCacheReader(args.cache)
    used = 0
    with cases_path.open("x") as stream:
        for index, ref in enumerate(reader):
            data = read_deepspec_features(ref, ["input_ids", "loss_mask"])
            anchors = select_anchors(data["loss_mask"].tolist(), args.k, args.anchors)
            if not anchors:
                continue
            for anchor in anchors:
                prefix = data["input_ids"][:anchor].tolist()
                pattern = f"prefix-{prefix_digest(prefix)}-*.pt"
                before = set(Path(args.dump_dir).glob(pattern))
                body = dict(model=args.model, prompt=prefix, temperature=0,
                            max_tokens=args.k + 1, ignore_eos=True)
                request = urllib.request.Request(args.url.rstrip("/") + "/v1/completions",
                    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=300) as response:
                    result = json.load(response)
                if not result.get("choices"):
                    raise RuntimeError(f"Completion failed: {result}")
                traces = sorted(set(Path(args.dump_dir).glob(pattern)) - before)
                if len(traces) != 1:
                    raise RuntimeError(f"Expected one first-prefill trace, found {len(traces)}. "
                        "Use eager serving without prefix caching/chunked prefill and ensure "
                        "the server and collector share --dump-dir.")
                trace = torch.load(traces[0], map_location="cpu", weights_only=True)
                if trace["prefix_token_ids"] != prefix or trace["anchor_position"] != anchor:
                    raise RuntimeError("Trace prefix/anchor mismatch")
                case = dict(cache_index=index, anchor=anchor, trace=str(traces[0].resolve()),
                            reference_anchor=int(data["input_ids"][anchor]),
                            reference_tokens=data["input_ids"][anchor + 1:anchor + 1 + args.k].tolist())
                stream.write(json.dumps(case) + "\n")
                stream.flush()
                print(f"Collected cache row {index}, anchor {anchor}", flush=True)
            used += 1
            if used >= args.records:
                break
    if not used:
        raise RuntimeError("No fully supervised response block found")
    print(cases_path)


def compare(args):
    import torch
    from specforge.modeling.auto import AutoDraftModel
    from specforge.modeling.target.offline_config import load_offline_target_config
    from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
    from specforge.runtime.data_plane.deepspec_cache import DeepSpecCacheReader, read_deepspec_features
    from specforge.algorithms.common.dflash_family_model import create_dflash_sdpa_mask

    cases_path = Path(args.cases)
    report_path = cases_path.with_name("comparison.json")
    if report_path.exists():
        raise FileExistsError(report_path)
    cases = [json.loads(line) for line in cases_path.read_text().splitlines() if line.strip()]
    if not cases:
        raise ValueError("Empty cases file")
    draft, loading = AutoDraftModel.from_pretrained(args.checkpoint, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16, output_loading_info=True)
    missing = [key for key in loading.get("missing_keys", []) if "embed_tokens" not in key]
    if missing or loading.get("unexpected_keys") or loading.get("mismatched_keys"):
        raise ValueError(f"Draft weights were not fully loaded: {loading}")
    draft = draft.to(args.device).eval()
    target_config = load_offline_target_config(args.target)
    target = TargetEmbeddingsAndHead.from_pretrained(args.target, config=target_config,
        embed_key="model.language_model.embed_tokens.weight", lm_head_key="lm_head.weight",
        device=args.device, dtype=torch.bfloat16)
    reader = DeepSpecCacheReader(args.cache)
    reader.validate_model(hidden_size=2560, target_layer_ids=[45, 46, 47],
                          target_model_path=args.target, target_config=target_config)
    required = {case["cache_index"] for case in cases}
    refs = {}
    for index, ref in enumerate(reader):
        if index in required:
            refs[index] = ref
        if len(refs) == len(required):
            break

    def metrics(left, right):
        left, right = left.float(), right.float()
        if left.shape != right.shape:
            raise ValueError(f"Feature shapes differ: {left.shape} vs {right.shape}")
        delta = left - right
        return dict(relative_l2=float(delta.norm() / right.norm().clamp_min(1e-12)),
                    max_abs=float(delta.abs().max()),
                    cosine=float(torch.nn.functional.cosine_similarity(left.flatten(), right.flatten(), dim=0)))

    reports = []
    with torch.inference_mode():
        for case in cases:
            trace = torch.load(case["trace"], map_location="cpu", weights_only=True)
            if Path(trace["draft_model"]).resolve() != Path(args.checkpoint).resolve():
                raise ValueError("Online/offline checkpoints differ")
            if not trace["sample_from_anchor"]:
                raise ValueError("This probe requires sample_from_anchor=True")
            data = read_deepspec_features(refs[case["cache_index"]], ["input_ids", "aux_hidden_state"])
            a, ids = case["anchor"], trace["draft_token_ids"]
            k = len(ids)
            if data["input_ids"][:a].tolist() != trace["prefix_token_ids"]:
                raise ValueError("Cache differs from collected prefix")
            if trace["query_positions"].tolist() != list(range(a, a + k)):
                raise ValueError("Unexpected online query positions")
            cache_aux = data["aux_hidden_state"][:a]
            online_aux = trace["aux_hidden"]
            anchor_ids = torch.tensor([trace["anchor_token_id"]], device=args.device)
            noise_ids = torch.full((1, k), draft.mask_token_id, device=args.device, dtype=torch.long)
            noise_ids[:, 0] = anchor_ids
            position_ids = torch.arange(a + k, device=args.device)[None]
            mask = create_dflash_sdpa_mask(torch.tensor([[a]], device=args.device),
                torch.ones((1, 1), dtype=torch.bool, device=args.device), a, k, args.device,
                context_window=getattr(draft, "context_window", None))

            def replay(aux):
                hidden = draft(position_ids=position_ids, attention_mask=mask,
                    noise_embedding=target.embed_tokens(noise_ids),
                    target_hidden=aux.to(args.device)[None])
                logits = target.lm_head(hidden)
                reranker = getattr(draft, "prefix_reranker", None)
                if reranker is not None:
                    tokens = reranker.sample(logits, hidden, anchor_ids, draft.markov_head, temperature=0.)
                else:
                    tokens, _ = draft.markov_head.sample_block_tokens(logits,
                        first_prev_token_ids=anchor_ids, hidden_states=hidden, temperature=0.)
                return hidden[0].cpu(), tokens[0].cpu().tolist()

            cache_hidden, cache_ids = replay(cache_aux)
            online_hidden, online_ids = replay(online_aux)
            entry = dict(cache_index=case["cache_index"], anchor=a,
                actual_anchor=trace["anchor_token_id"], reference_anchor=case["reference_anchor"],
                aux_difference=metrics(cache_aux, online_aux),
                hidden_with_cache_aux=metrics(cache_hidden, trace["draft_hidden"]),
                hidden_with_online_aux=metrics(online_hidden, trace["draft_hidden"]),
                vllm_tokens=ids, offline_cache_aux_tokens=cache_ids, offline_online_aux_tokens=online_ids,
                cache_aux_token_matches=[x == y for x, y in zip(cache_ids, ids)],
                online_aux_token_matches=[x == y for x, y in zip(online_ids, ids)])
            reports.append(entry)
            print(json.dumps(entry), flush=True)
    report_path.write_text(json.dumps(reports, indent=2) + "\n")
    print(report_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    c = commands.add_parser("collect")
    c.add_argument("--cache", required=True)
    c.add_argument("--dump-dir", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--url", default="http://127.0.0.1:8000")
    c.add_argument("--model", default="qwen38-flash-next")
    c.add_argument("--records", type=int, default=2)
    c.add_argument("--anchors", type=int, default=3)
    c.add_argument("--k", type=int, default=7)
    p = commands.add_parser("compare")
    p.add_argument("--cache", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--device", default="cuda")
    for name in ("inspect-scores", "verify-target"):
        extra = commands.add_parser(name)
        extra.add_argument("--comparison", required=True)
        extra.add_argument("--cases", required=True)
        extra.add_argument("--output", required=True)
        extra.add_argument("--all-blocks", action="store_true")
        if name == "inspect-scores":
            extra.add_argument("--checkpoint", required=True)
            extra.add_argument("--target", required=True)
            extra.add_argument("--device", default="cuda")
        else:
            extra.add_argument("--url", default="http://127.0.0.1:8000")
            extra.add_argument("--model", default="qwen38-flash-next")
    args = parser.parse_args()
    if args.command == "collect":
        if min(args.records, args.anchors, args.k) <= 0:
            parser.error("records, anchors and k must be positive")
        collect(args)
    elif args.command == "compare":
        compare(args)
    elif args.command == "inspect-scores":
        inspect_scores(args)
    else:
        verify_target(args)


if __name__ == "__main__":
    main()
