"""Paired, fixed-prefix SpecForge diagnostics for full-vocabulary Qwen3 DSpark.

Run from the repository root with PYTHONPATH=.  --draft takes NAME CONFIG
CHECKPOINT (training_state.pt / step directory, or a SpecForge HF export).
Use the same saved GSM8K responses.jsonl for every draft. For exact replay
provide JSONL rows with prompt_token_ids and completion_token_ids. Text-only
benchmark rows are reconstructed using their saved request and target tokenizer.
A completed run can be summarized on CPU using --summarize OUTPUT_DIR.
"""

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import os
import zipfile
from pathlib import Path

import torch


def read_json(path):
    return json.loads(path.read_text())


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".pt":
        torch.save(value, tmp)
    else:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def fingerprint(path):
    """Cheap identity for large local weights, including HF shards."""
    path = Path(path).resolve(strict=True)
    files = (
        [path]
        if path.is_file()
        else sorted(
            p
            for p in path.iterdir()
            if p.is_file()
            and p.suffix in {".json", ".safetensors", ".pt", ".bin", ".model"}
        )
    )
    return [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in files]


def token_sequence(row, tokenizer):
    request = row.get("request", {})
    response = row.get("response", {})
    choice = (response.get("choices") or [{}])[0]
    prompt = row.get("prompt_token_ids", response.get("prompt_token_ids"))
    completion = row.get("completion_token_ids", choice.get("token_ids"))
    sources = []
    if prompt is None:
        if "messages" in request:
            prompt = tokenizer.apply_chat_template(
                request["messages"],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
                **request.get("chat_template_kwargs", {}),
            )
        elif isinstance(request.get("prompt"), str):
            prompt = tokenizer.encode(request["prompt"], add_special_tokens=True)
        else:
            raise ValueError("Expected token IDs or a GSM8K --save-details request")
        sources.append("prompt_from_text")
    if completion is None:
        if row.get("reasoning") or choice.get("message", {}).get("reasoning_content"):
            raise ValueError("A reasoning response needs original completion_token_ids")
        content = row.get("content", choice.get("text"))
        if not isinstance(content, str):
            raise ValueError("Missing saved completion text or token IDs")
        completion = tokenizer.encode(content, add_special_tokens=False)
        sources.append("completion_from_text")
    for ids in (prompt, completion):
        if not isinstance(ids, list) or any(type(x) is not int or x < 0 for x in ids):
            raise ValueError("Expected lists of nonnegative integer token IDs")
    return prompt, completion, "+".join(sources) or "exact_ids"


def select_rows(args, tokenizer, width):
    rows = [
        json.loads(line)
        for line in args.responses.read_text().splitlines()
        if line.strip()
    ]
    indices = list(range(len(rows)))
    random.Random(args.seed).shuffle(indices)
    samples = []
    for i in indices:
        row = rows[i]
        if row.get("error") or row.get("http_status", 200) != 200:
            continue
        prompt, reply, source = token_sequence(row, tokenizer)
        ids = (prompt + reply)[: args.max_length]
        # Score inside the response. All anchors have width valid next tokens.
        first, last = len(prompt), len(ids) - width - 1
        if not prompt or first > last:
            continue
        n = min(args.anchors, last - first + 1)
        anchors = (
            [int((first + last) // 2)]
            if n == 1
            else [first + round(k * (last - first) / (n - 1)) for k in range(n)]
        )
        samples.append(
            {
                "row": i,
                "question": row.get("question_index", i),
                "ids": ids,
                "anchors": anchors,
                "prompt_length": len(prompt),
                "source": source,
                "reply_length": len(reply),
                "truncated": len(prompt) + len(reply) > len(ids),
                "reported_prompt_tokens": row.get("response", {})
                .get("usage", {})
                .get("prompt_tokens"),
                "reported_completion_tokens": row.get("completion_tokens"),
            }
        )
        if len(samples) == args.questions:
            break
    if not samples:
        raise ValueError(
            "No valid response has a complete draft block within --max-length"
        )
    return samples


def perfectblend_rows(args, tokenizer, width):
    # Index byte offsets so sampling a large training JSONL never loads it all into RAM.
    with args.perfectblend.open("rb") as stream:
        offsets = []
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)
        random.Random(args.seed).shuffle(offsets)
        samples = []
        for offset in offsets:
            stream.seek(offset)
            row = json.loads(stream.readline())
            if "input_ids" in row:
                ids, mask = row["input_ids"], row["loss_mask"]
                source = "training_ids_and_mask"
            elif "prompt_token_ids" in row and "completion_token_ids" in row:
                prompt, reply, source = token_sequence(row, tokenizer)
                ids, mask = prompt + reply, [0] * len(prompt) + [1] * len(reply)
            else:
                from specforge.data.preprocessing import preprocess_conversations
                from specforge.data.template import TEMPLATE_REGISTRY

                conversation = row.get("conversations", row.get("messages"))
                if conversation is None:
                    raise ValueError(
                        f"Unsupported PerfectBlend format at byte {offset}"
                    )
                if not conversation:
                    continue
                tools = row.get("tools") or []
                if isinstance(tools, str):
                    tools = json.loads(tools)
                extras = {
                    k: [v]
                    for k, v in row.items()
                    if k not in {"id", "conversations", "messages", "tools"}
                }
                batch = preprocess_conversations(
                    tokenizer,
                    [conversation],
                    TEMPLATE_REGISTRY.get(args.perfectblend_template),
                    args.max_length,
                    tools=[tools],
                    **extras,
                )
                ids = batch["input_ids"][0].flatten().tolist()
                mask = batch["loss_mask"][0].flatten().tolist()
                source = "training_chat_template"
            if (
                len(ids) != len(mask)
                or any(type(x) is not int or x < 0 for x in ids)
                or any(x not in (0, 1) for x in mask)
            ):
                raise ValueError(f"Invalid training IDs or loss_mask at byte {offset}")
            ids, mask = ids[: args.max_length], mask[: args.max_length]
            candidates = [
                a for a in range(len(ids) - width) if all(mask[a : a + width + 1])
            ]
            n = min(args.anchors, len(candidates))
            if not n:
                continue
            anchors = (
                [candidates[(len(candidates) - 1) // 2]]
                if n == 1
                else [
                    candidates[round(k * (len(candidates) - 1) / (n - 1))]
                    for k in range(n)
                ]
            )
            samples.append(
                {
                    "row": f"pb-{offset}",
                    "dataset": "perfectblend",
                    "source": source,
                    "ids": ids,
                    "loss_mask": mask,
                    "anchors": anchors,
                }
            )
            if len(samples) == args.questions:
                break
    if not samples:
        raise ValueError("PerfectBlend has no full response block within --max-length")
    return samples


def draft_state(path):
    return path / "training_state.pt" if path.is_dir() else path


def load_training_state(path):
    # SpecForge also writes legacy torch.save files, which cannot be memory-mapped.
    return torch.load(
        path, map_location="cpu", weights_only=False, mmap=zipfile.is_zipfile(path)
    )


def load_draft(config_path, path, device):
    from specforge.export.checkpoint_io import materialize_draft
    from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig

    config = read_json(config_path)
    opts = config["dflash_config"]
    state_file = draft_state(path)
    if state_file.is_file():
        state = load_training_state(state_file)
        expected = {
            "dspark_target_layer_ids": opts["target_layer_ids"],
            "dspark_block_size": config["block_size"],
            "dspark_mask_token_id": opts["mask_token_id"],
        }
        for key, value in expected.items():
            actual = state.get(key)
            if key == "dspark_target_layer_ids" and actual is not None:
                actual = list(actual)
            if actual is not None and actual != value:
                raise ValueError(f"{path}: {key} checkpoint={actual} config={value}")
        model = materialize_draft(state, str(config_path))
        contract = {
            key: state.get(key) for key in (*expected, "dspark_engram", "global_step")
        }
    else:
        if (
            not (path / "config.json").is_file()
            or read_json(path / "config.json") != config
        ):
            raise ValueError(
                "For a SpecForge HF export use its own config.json as CONFIG"
            )
        cfg = AutoDraftModelConfig.from_file(str(config_path))
        model, info = AutoDraftModel.from_pretrained(
            str(path),
            config=cfg,
            output_loading_info=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            if info.get(key):
                raise ValueError(f"{path}: {key}={info[key]}")
        contract = {"source": "HF export"}
    if getattr(model.markov_head, "markov_head_type", None) != opts["markov_head_type"]:
        raise ValueError(f"{path}: loaded Markov head disagrees with config")
    contract["backbone_rope_dtype"] = str(model.rotary_emb.inv_freq.dtype)
    contract["resolved_head"] = model.markov_head.markov_head_type
    model.config._attn_implementation = "sdpa"
    return model.eval().to(device), contract


def capture_target(target, sample, layers, device):
    ids = torch.tensor([sample["ids"]], device=device)
    captured = {}
    handles = []

    # Last decoder-layer output can differ from hidden_states[-1] (final RMSNorm).
    def capture(index):
        def hook(module, inputs, output):
            captured[index] = (
                output[0] if isinstance(output, tuple) else output
            ).detach()

        return hook

    try:
        for index in layers:
            handles.append(
                target.model.layers[index].register_forward_hook(capture(index))
            )
        final = target.model(input_ids=ids, use_cache=False).last_hidden_state
    finally:
        for handle in handles:
            handle.remove()
    return {
        "context": {i: captured[i][0].cpu() for i in layers},
        "final": final[0].cpu(),
    }


def predict(wrapper, ids, context, anchors):
    model = wrapper.draft_model
    keep = torch.ones_like(anchors, dtype=torch.bool)
    wants_context = getattr(model, "requires_markov_context", False)
    wants_tokens = getattr(model, "requires_markov_token_history", False)
    output = wrapper._forward_draft_blocks(
        ids,
        context,
        torch.ones_like(ids),
        anchor_positions=anchors,
        block_keep_mask=keep,
        **({"return_context": True} if wants_context else {}),
    )
    hidden = output[2].reshape(1, anchors.numel(), model.block_size, -1)
    pos = anchors.unsqueeze(-1) + torch.arange(model.block_size, device=ids.device)
    safe = pos.clamp_max(ids.size(1) - 1)
    prev = ids[0, safe]
    extra = {}
    buckets = None
    if getattr(model, "requires_engram_history", False):
        buckets = model.markov_head.memory.bucket_ids(ids)[0, safe]
        extra["engram_bucket_ids"] = buckets
    if wants_context or wants_tokens:
        extra.update(
            query_position_ids=pos,
            anchor_positions=anchors,
            block_keep_mask=keep,
            attention_backend=wrapper.attention_backend,
        )
        if wants_tokens:
            extra["history_token_ids"] = ids
        else:
            extra["target_context"] = output[3]
    base = wrapper.lm_head(hidden)
    final = model.apply_logits_head(
        base, prev_token_ids=prev, hidden_states=hidden, **extra
    )
    return base[0], final[0], hidden[0], buckets


def measure(target, base, final, labels):
    p = target.float().softmax(-1)
    p_id = target.argmax(-1)
    metrics = {}
    for name, logits in (("base", base), ("final", final)):
        q = logits.float().softmax(-1)
        top = logits.argmax(-1)
        # Preserve dimension [anchor, position] until reporting.
        metrics[name + "_overlap"] = torch.minimum(p, q).sum(-1)
        metrics[name + "_target_top1"] = (top == p_id).float()
        metrics[name + "_label_top1"] = (top == labels).float()
        metrics[name + "_label_nll"] = (
            -logits.float().log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
        )
    metrics["head_overlap_gain"] = metrics["final_overlap"] - metrics["base_overlap"]
    metrics["head_flips_top1"] = (final.argmax(-1) != base.argmax(-1)).float()
    metrics["effective_bias_rms"] = (
        (final.float() - base.float()).square().mean(-1).sqrt()
    )
    metrics["target_margin"] = -target.float().topk(2).values.diff(dim=-1).squeeze(-1)
    metrics["target_hits_label"] = (labels == p_id).float()
    values = torch.stack(list(metrics.values())).cpu()
    if not values.isfinite().all():
        raise FloatingPointError(
            "Non-finite target or draft metrics; inspect this checkpoint"
        )
    return dict(zip(metrics, values.unbind(), strict=True))


def rope_probe(wrapper, ids, context, anchors, target, original):
    # Rebuild from config: upcasting a previously rounded frequency cannot
    # recover the full-precision value. Change only the backbone RoPE module.
    model = wrapper.draft_model
    saved = model.rotary_emb
    alt_dtype = (
        torch.float32 if saved.inv_freq.dtype != torch.float32 else torch.bfloat16
    )
    fresh = type(saved)(model.config).to(device=ids.device, dtype=alt_dtype)
    try:
        model.rotary_emb = fresh
        _, alternate, _, _ = predict(wrapper, ids, context, anchors)
    finally:
        model.rotary_emb = saved
    p = target[:, 0].float().softmax(-1)
    before = original[:, 0].float()
    after = alternate[:, 0].float()
    change = torch.minimum(p, after.softmax(-1)).sum(-1) - torch.minimum(
        p, before.softmax(-1)
    ).sum(-1)
    return {
        "original_dtype": str(saved.inv_freq.dtype),
        "alternate_dtype": str(alt_dtype),
        "max_frequency_difference": (fresh.inv_freq.float() - saved.inv_freq.float())
        .abs()
        .max()
        .item(),
        "first_overlap_delta_pp": (100 * change).tolist(),
        "first_top1_changed": (after.argmax(-1) != before.argmax(-1)).tolist(),
        "first_logit_max_abs": (after - before).abs().amax(-1).tolist(),
    }


def diagnose_question(wrapper, capture, sample, sample_path, probe, device):
    model = wrapper.draft_model
    context = torch.cat([capture["context"][i] for i in model.target_layer_ids], -1)[
        None
    ].to(device)
    ids = torch.tensor([sample["ids"]], device=device)
    anchors = torch.tensor([sample["anchors"]], device=device)
    pos = anchors[0, :, None] + torch.arange(model.block_size, device=device)
    labels = ids[0, pos + 1]
    target = wrapper.lm_head(capture["final"][pos.cpu()].to(device))
    projection_trace = {}

    def record_projection(module, inputs, output):
        if not projection_trace:
            projection_trace["markov_latent"] = (
                inputs[0].reshape(-1, inputs[0].shape[-1])[0].detach().cpu().clone()
            )
            projection_trace["markov_bias"] = (
                output.reshape(-1, output.shape[-1])[0].detach().cpu().clone()
            )

    hook = (
        model.markov_head.markov_w2.register_forward_hook(record_projection)
        if probe
        else None
    )
    try:
        base, final, hidden, buckets = predict(wrapper, ids, context, anchors)
    finally:
        if hook is not None:
            hook.remove()
    metrics = measure(target, base, final, labels)
    per_position = {k: v.mean(0).tolist() for k, v in metrics.items()}
    top = {}
    for key, logits in (("target", target), ("base", base), ("final", final)):
        values, indices = logits[:, 0].float().topk(5)
        top[key] = {"ids": indices.tolist(), "logits": values.tolist()}
    record = {
        "row": sample["row"],
        "anchors": sample["anchors"],
        "per_position": per_position,
        "first": {k: v[:, 0].tolist() for k, v in metrics.items()},
        "first_top5": top,
        "teacher_forced_tau": (1 + metrics["final_overlap"].cumprod(-1).sum(-1))
        .mean()
        .item(),
    }
    if probe:
        # Compare identical shapes, one fixed anchor. Perturb only future IDs and
        # their auxiliary features. This specifically tests draft-side future leakage.
        a = sample["anchors"][0]
        anchor = anchors[:, :1]
        ref_base, ref, _, _ = predict(wrapper, ids, context, anchor)
        changed_ids, changed_context = ids.clone(), context.clone()
        changed_ids[:, a + 1 :] = (
            changed_ids[:, a + 1 :] + 1
        ) % model.config.vocab_size
        changed_context[:, a + 1 :] = 0
        alt_base, alt, _, _ = predict(wrapper, changed_ids, changed_context, anchor)
        record["future_probe"] = {
            "anchor": a,
            "base_max_abs": (ref_base[0, 0].float() - alt_base[0, 0].float())
            .abs()
            .max()
            .item(),
            "final_max_abs": (ref[0, 0].float() - alt[0, 0].float()).abs().max().item(),
            "final_top1_changed": bool(ref[0, 0].argmax() != alt[0, 0].argmax()),
        }
        record["backbone_rope_probe"] = rope_probe(
            wrapper, ids, context, anchors, target, final
        )
        # Full first-position vectors for later CPU / vLLM comparisons. The bias
        # here is the effective change after the model's low-precision addition.
        trace = {
            "row": sample["row"],
            "prefix_ids": ids[0, : a + 1].cpu(),
            "anchor": a,
            "target_logits": target[0, 0].cpu().clone(),
            "base_logits": base[0, 0].cpu().clone(),
            "final_logits": final[0, 0].cpu().clone(),
            "hidden": hidden[0, 0].cpu().clone(),
            "effective_bias": (final[0, 0].float() - base[0, 0].float()).cpu(),
        }
        trace.update(projection_trace)
        if buckets is not None:
            trace["bucket_ids"] = buckets[0, 0, 0].cpu()
            trace["memory_correction"] = (
                model.markov_head.memory(buckets[:, :1, :1], hidden[None, :1, :1])
                .flatten()
                .cpu()
            )
        save(sample_path.with_suffix(".pt"), trace)
    save(sample_path, record)


def load_live(out, source, sample, layers, width):
    data = torch.load(
        out / "aux-comparison" / source / f"row-{sample['row']}.pt",
        map_location="cpu",
        weights_only=True,
    )
    if data["ids"] != sample["ids"] or data["layers"] != layers:
        raise ValueError(
            f"{source}: saved tokens/layers disagree for row {sample['row']}"
        )
    if data["aux"].shape != (len(sample["ids"]), width * len(layers)):
        raise ValueError(f"{source}: invalid aux shape")
    return data


def compare_positions(trainer, vllm, width):
    import torch.nn.functional as F

    report = {}
    for layer, left, right in zip(
        trainer["layers"],
        trainer["aux"].split(width, -1),
        vllm["aux"].split(width, -1),
        strict=True,
    ):
        a, b = left.float(), right.float()
        error = b - a
        rms = error.square().mean(-1).sqrt()
        report[str(layer)] = {
            "rms": rms.tolist(),
            "relative_rms": (
                rms / a.square().mean(-1).sqrt().clamp_min(1e-12)
            ).tolist(),
            "max_abs": error.abs().amax(-1).tolist(),
            "cosine": F.cosine_similarity(a, b).tolist(),
            "vllm_vs_previous_trainer_cosine": [None]
            + F.cosine_similarity(b[1:], a[:-1]).tolist(),
            "vllm_vs_next_trainer_cosine": F.cosine_similarity(b[:-1], a[1:]).tolist()
            + [None],
        }
    return {"token_ids": trainer["ids"], "layers_by_position": report}


def replay_summary(out, samples, names, sources=("trainer", "vllm")):
    rows = []
    for sample in samples:
        row = sample["row"]
        scores = {}
        for name in names:
            for source in sources:
                directory = (
                    out / name
                    if source == "trainer"
                    else out / "aux-comparison" / "replay" / name
                )
                path = directory / f"row-{row}.json"
                if path.exists():
                    record = read_json(path)
                    if record["anchors"] != sample["anchors"] or record["row"] != row:
                        raise ValueError(f"{path}: unpaired replay")
                    scores[source, name] = record["per_position"]["final_overlap"][0]
        if len(scores) == len(names) * len(sources):
            rows.append(scores)
    result = {"paired_questions": len(rows), "first_overlap": {}, "comparisons": {}}
    if not rows:
        return result

    def mean(values):
        return sum(values) / len(values)

    for source in sources:
        result["first_overlap"][source] = {
            n: mean([r[source, n] for r in rows]) for n in names
        }
    picks = torch.randint(
        len(rows), (2000, len(rows)), generator=torch.Generator().manual_seed(42)
    )

    def estimate(values):
        x = torch.tensor(values, dtype=torch.float64) * 100
        ci = torch.quantile(x[picks].mean(1), x.new_tensor([0.025, 0.975]))
        return {
            "delta_pp": x.mean().item(),
            "ci95_pp": ci.tolist() if len(x) > 1 else None,
        }

    for name in names[1:]:
        differences = {s: [r[s, name] - r[s, names[0]] for r in rows] for s in sources}
        result["comparisons"][name] = {
            "reference": names[0],
            **{s: estimate(x) for s, x in differences.items()},
        }
        if "vllm" in sources:
            tr, vl = differences["trainer"], differences["vllm"]
            result["comparisons"][name].update(
                vllm_minus_trainer_interaction=estimate(
                    [v - t for t, v in zip(tr, vl, strict=True)]
                ),
                observed_rank_reversal=mean(tr) * mean(vl) < 0,
            )
    return result


def summarize(out):
    manifest = read_json(out / "manifest.json")
    samples = read_json(out / "prefixes.json")
    names = list(manifest["drafts"])
    records = {name: {} for name in names}
    summary = {"planned_questions": len(samples), "models": {}, "paired": {}}
    for name in names:
        for s in samples:
            path = out / name / f"row-{s['row']}.json"
            if path.exists():
                records[name][s["row"]] = read_json(path)
        rows = list(records[name].values())
        if not rows:
            continue
        by_position = {
            k: torch.tensor([r["per_position"][k] for r in rows])
            .double()
            .mean(0)
            .tolist()
            for k in rows[0]["per_position"]
        }
        probes = [r["future_probe"] for r in rows if "future_probe" in r]
        summary["models"][name] = {
            "questions": len(rows),
            "anchors": sum(len(r["anchors"]) for r in rows),
            "question_mean_by_position": by_position,
            "question_mean_teacher_forced_tau": sum(
                r["teacher_forced_tau"] for r in rows
            )
            / len(rows),
            "future_probes": probes,
            "rope_probes": [
                r["backbone_rope_probe"] for r in rows if "backbone_rope_probe" in r
            ],
        }
    baseline = names[0]
    for name in names[1:]:
        common = sorted(records[baseline].keys() & records[name].keys())
        if not common:
            continue
        # Resample questions as clusters; nearby anchors are correlated.
        rng = torch.Generator().manual_seed(42)
        picks = torch.randint(len(common), (2000, len(common)), generator=rng)
        diffs = {}
        for key in (
            "final_overlap",
            "base_overlap",
            "head_overlap_gain",
            "final_target_top1",
            "final_label_top1",
        ):
            delta = torch.tensor(
                [
                    records[name][i]["per_position"][key][0]
                    - records[baseline][i]["per_position"][key][0]
                    for i in common
                ],
                dtype=torch.float64,
            )
            ci = torch.quantile(
                delta[picks].mean(1), torch.tensor([0.025, 0.975], dtype=delta.dtype)
            )
            diffs[key] = {
                "delta_pp": 100 * delta.mean().item(),
                "ci95_pp": (100 * ci).tolist() if len(common) > 1 else None,
            }
        summary["paired"][name] = {
            "reference": baseline,
            "questions": len(common),
            "first_position": diffs,
        }
    summary["complete"] = all(len(r) == len(samples) for r in records.values())
    if "aux_comparison" in manifest:
        live = replay_summary(out, samples, names)
        summary["aux_comparison"] = live
        summary["complete"] &= live["paired_questions"] == len(samples)
        if "perfectblend" in manifest:
            pb_samples = read_json(out / "perfectblend-prefixes.json")
            pb = replay_summary(out, pb_samples, names, sources=("trainer",))
            summary["perfectblend"] = pb
            summary["complete"] &= pb["paired_questions"] == len(pb_samples)
            if summary["complete"]:
                summary["first_negative_stage"] = {}
                for name in names[1:]:
                    stages = (
                        ("perfectblend_trainer", pb["comparisons"][name]["trainer"]),
                        ("gsm8k_trainer", live["comparisons"][name]["trainer"]),
                        ("gsm8k_vllm", live["comparisons"][name]["vllm"]),
                    )
                    summary["first_negative_stage"][name] = next(
                        (stage for stage, item in stages if item["delta_pp"] < 0), None
                    )
        print(json.dumps(live, indent=2), flush=True)
    save(out / "summary.json", summary)
    print(json.dumps(summary["paired"], indent=2), flush=True)
    print(
        f"Complete: {summary['complete']}. Report: {out / 'summary.json'}", flush=True
    )
    return summary


def run(args):
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from transformers import (
        __version__ as hf_version,
    )

    from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel

    out = args.output
    configs = {}
    drafts = {}
    for name, cfg_path, path in args.draft:
        if (
            not re.fullmatch(r"[A-Za-z0-9_-]+", name)
            or name in drafts
            or name in {"target-cache", "aux-comparison"}
        ):
            raise ValueError(f"Duplicate or reserved/invalid model label: {name}")
        path, cfg_path = Path(path).resolve(), Path(cfg_path).resolve()
        config = read_json(cfg_path)
        if config["model_type"] != "qwen3" or config.get("draft_vocab_size") not in (
            None,
            config["vocab_size"],
        ):
            raise ValueError("This diagnostic requires full-vocabulary Qwen3 DSpark")
        if not (draft_state(path).is_file() or (path / "config.json").is_file()):
            raise ValueError(f"Expected a step checkpoint or HF export: {path}")
        configs[name] = config
        drafts[name] = {
            "config": str(cfg_path),
            "config_values": config,
            "path": str(path),
            "weights": fingerprint(
                draft_state(path) if draft_state(path).is_file() else path
            ),
        }
    widths = {cfg["block_size"] for cfg in configs.values()}
    if len(widths) != 1:
        raise ValueError("Paired runs require the same block size")
    layers = sorted(
        {
            i
            for cfg in configs.values()
            for i in cfg["dflash_config"]["target_layer_ids"]
        }
    )
    target_config = read_json(args.target / "config.json")
    if target_config["model_type"] != "qwen3" or any(
        c["vocab_size"] != target_config["vocab_size"] for c in configs.values()
    ):
        raise ValueError(
            "Target must be text Qwen3 with the same full vocabulary as the drafts"
        )
    if min(layers) < 0 or max(layers) >= target_config["num_hidden_layers"]:
        raise ValueError("Configured capture layers outside target backbone")
    try:
        commit = subprocess.check_output(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parents[1]),
                "rev-parse",
                "HEAD",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {
        "target": fingerprint(args.target),
        "target_dir": str(args.target.resolve()),
        "drafts": drafts,
        "reference": next(iter(drafts)),
        "responses": str(args.responses.resolve()),
        "responses_sha256": hashlib.sha256(args.responses.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "questions": args.questions,
        "anchors": args.anchors,
        "max_length": args.max_length,
        "seed": args.seed,
        "probe_questions": args.probe_questions,
        "git_commit": commit,
        "device": args.device,
        "dtype": "bfloat16",
        "attention_backend": "sdpa",
        "torch": str(torch.__version__),
        "transformers": hf_version,
        "cuda": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(args.device)
            if torch.device(args.device).type == "cuda"
            else None
        ),
    }
    if args.perfectblend:
        with args.perfectblend.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        manifest["perfectblend"] = {
            "path": str(args.perfectblend.resolve()),
            "sha256": digest,
            "template": args.perfectblend_template,
        }
    if args.expected_step is not None:
        manifest["expected_step"] = args.expected_step
        for spec in drafts.values():
            file = draft_state(Path(spec["path"]))
            step = load_training_state(file).get("global_step") if file.is_file() else None
            if step is not None and int(step) != args.expected_step:
                raise ValueError(
                    f"{file}: global_step={step}, expected {args.expected_step}"
                )
            spec["verified_step"] = int(step) if step is not None else None
            if step is None:
                print(
                    f"Checkpoint step unavailable in {file}; confirm step {args.expected_step} from export provenance",
                    flush=True,
                )
    worker = Path(__file__).with_name("dspark_aux_capture.py")
    if args.compare_aux:
        if any(
            c["dflash_config"]["target_layer_ids"] != layers for c in configs.values()
        ):
            raise ValueError(
                "Live replay requires the same ordered, ascending capture layers"
            )
        if not (args.compare_aux / "config.json").is_file():
            raise ValueError(
                "--compare-aux requires a vLLM-loadable DSpark export directory"
            )
        manifest["aux_comparison"] = {
            "export": str(args.compare_aux.resolve()),
            "sglang_attention_backend": args.sglang_attention_backend,
            "vllm_eager": args.vllm_eager,
            "weights": fingerprint(args.compare_aux),
            "layers": layers,
            "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
            "vllm_python": args.vllm_python,
            "batch_invariant": os.getenv("VLLM_BATCH_INVARIANT"),
            "reference_distribution": "trainer post-norm last hidden + frozen target lm_head, T=1",
            "primary_feature_source": "trainer",
            "capture": "teacher-forced prefill, TP=1",
        }
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    # JSON round trip canonicalizes tuples in file fingerprints.
    if manifest_path.exists():
        if read_json(manifest_path) != json.loads(json.dumps(manifest)):
            raise ValueError(
                "Run inputs, code, settings or weights changed. Use a new --output."
            )
    elif any(out.iterdir()):
        raise ValueError(f"Output directory has unrelated files: {out}")
    else:
        save(manifest_path, manifest)
    prefixes = out / "prefixes.json"
    if not prefixes.exists():
        tokenizer = AutoTokenizer.from_pretrained(args.target)
        save(prefixes, select_rows(args, tokenizer, next(iter(widths))))
    samples = read_json(prefixes)
    pb_samples = []
    if args.perfectblend:
        pb_path = out / "perfectblend-prefixes.json"
        if not pb_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(args.target)
            save(pb_path, perfectblend_rows(args, tokenizer, next(iter(widths))))
        pb_samples = read_json(pb_path)
    all_samples = pb_samples + samples
    cache = out / "target-cache"
    cache.mkdir(exist_ok=True)

    def replay_path(name, sample):
        return out / "aux-comparison" / "replay" / name / f"row-{sample['row']}.json"

    def finished(name, sample):
        return (out / name / f"row-{sample['row']}.json").exists() and (
            not args.compare_aux
            or sample.get("dataset") == "perfectblend"
            or replay_path(name, sample).exists()
        )

    if all(finished(name, s) for name in drafts for s in all_samples):
        summarize(out)
        return
    if args.compare_aux:
        for source, executable in (
            ("trainer", sys.executable),
            ("vllm", args.vllm_python),
        ):
            subprocess.run(
                [executable, str(worker), source, str(out.resolve())], check=True
            )
        detail = out / "aux-comparison" / "positions"
        detail.mkdir(exist_ok=True)
        width = read_json(args.target / "config.json")["hidden_size"]
        for sample in samples:
            path = detail / f"row-{sample['row']}.json"
            if not path.exists():
                tr, vl = [
                    load_live(out, source, sample, layers, width)
                    for source in ("trainer", "vllm")
                ]
                save(path, compare_positions(tr, vl, width))
        from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead

        frozen = TargetEmbeddingsAndHead.from_pretrained(
            str(args.target), device=args.device
        )
        projection, embedding = frozen.lm_head, frozen.embed_tokens
    else:
        target = (
            AutoModelForCausalLM.from_pretrained(
                args.target, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
            )
            .eval()
            .to(args.device)
        )
        projection = target.get_output_embeddings()
        embedding = target.get_input_embeddings()
        for k, sample in enumerate(samples):
            dest = cache / f"row-{sample['row']}.pt"
            if not dest.exists():
                save(dest, capture_target(target, sample, layers, args.device))
            if k % 10 == 0:
                print(f"Target cache {k + 1}/{len(samples)}", flush=True)
        del target  # Keep only the frozen embedding and output projection on device.
    for name, spec in drafts.items():
        directory = out / name
        directory.mkdir(exist_ok=True)
        pending = [(k, s) for k, s in enumerate(all_samples) if not finished(name, s)]
        if not pending:
            continue
        model, contract = load_draft(
            Path(spec["config"]), Path(spec["path"]), args.device
        )
        wrapper = OnlineDSparkModel(
            model,
            projection,
            embedding,
            model.mask_token_id,
            block_size=model.block_size,
            attention_backend="sdpa",
        ).eval()
        save(directory / "contract.json", contract)
        for k, sample in pending:
            if args.compare_aux:
                tr = load_live(out, "trainer", sample, layers, width)
                paths = [("trainer", directory / f"row-{sample['row']}.json")]
                if sample.get("dataset") != "perfectblend":
                    paths.append(("vllm", replay_path(name, sample)))
                for source, path in paths:
                    if path.exists():
                        continue
                    path.parent.mkdir(parents=True, exist_ok=True)
                    data = (
                        tr
                        if source == "trainer"
                        else load_live(out, source, sample, layers, width)
                    )
                    capture = {
                        "context": dict(
                            zip(layers, data["aux"].split(width, -1), strict=True)
                        ),
                        "final": tr["final"],
                    }
                    diagnose_question(
                        wrapper,
                        capture,
                        sample,
                        path,
                        source == "trainer"
                        and 0 <= k - len(pb_samples) < args.probe_questions,
                        args.device,
                    )
                continue
            capture = torch.load(
                cache / f"row-{sample['row']}.pt", map_location="cpu", weights_only=True
            )
            diagnose_question(
                wrapper,
                capture,
                sample,
                directory / f"row-{sample['row']}.json",
                k < args.probe_questions,
                args.device,
            )
            if k % 10 == 0:
                print(f"{name} {k + 1}/{len(samples)}", flush=True)
        del wrapper, model
    summarize(out)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--target", type=Path, help="Local Qwen3 target/tokenizer directory"
    )
    parser.add_argument(
        "--responses", type=Path, help="Shared GSM8K responses.jsonl or token-ID JSONL"
    )
    parser.add_argument(
        "--draft", nargs=3, action="append", metavar=("NAME", "CONFIG", "CHECKPOINT")
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New directory or an interrupted run with identical arguments",
    )
    parser.add_argument(
        "--summarize",
        type=Path,
        help="Rebuild a (possibly partial) report without models or a GPU",
    )
    parser.add_argument(
        "--compare-aux",
        type=Path,
        metavar="VLLM_DSPARK_EXPORT",
        help="Run private patched SGLang/Mooncake and vLLM V2 sequentially, then cross-replay",
    )
    parser.add_argument(
        "--vllm-python",
        default=sys.executable,
        help="Python from the DSpark vLLM environment",
    )
    parser.add_argument(
        "--sglang-attention-backend",
        default="flashinfer",
        help="Match the online teacher attention backend in the training launch plan",
    )
    parser.add_argument(
        "--vllm-eager",
        action="store_true",
        help="Disable vLLM compilation and graphs for capture (default: engine settings)",
    )
    parser.add_argument(
        "--perfectblend",
        type=Path,
        help="Open PerfectBlend training JSONL for trainer-only replay",
    )
    parser.add_argument("--perfectblend-template", default="qwen")
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--questions", type=int, default=64)
    parser.add_argument("--anchors", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--probe-questions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.perfectblend and not args.compare_aux:
        parser.error("--perfectblend requires --compare-aux")
    if args.summarize:
        summarize(args.summarize)
        return
    if (
        not all([args.target, args.responses, args.output, args.draft])
        or len(args.draft) < 2
    ):
        parser.error(
            "Supply --target, --responses, --output and at least two --draft triples"
        )
    if (
        min(args.questions, args.anchors, args.max_length) < 1
        or args.probe_questions < 0
    ):
        parser.error("Sizes must be positive and probe count nonnegative")
    with torch.inference_mode():
        run(args)


if __name__ == "__main__":
    main()
