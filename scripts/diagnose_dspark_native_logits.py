"""Append native V2 DSpark first-position comparisons to a completed aux run."""

import argparse
import hashlib
import inspect
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import torch

from diagnose_dspark_prefixes import (
    draft_state,
    fingerprint,
    load_draft,
    load_live,
    predict,
    read_json,
    save,
)


def file_id(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_anchor(
    engine, proposer, params, ids, a, reference, layers, vocab, resolve_layers
):
    """Prefill x[:a], force target bonus x[a], observe draft prediction of x[a+1]."""
    original = proposer.propose
    signature = inspect.signature(original)
    trace = {}

    def observe(self, *args, **kwargs):
        call = signature.bind(self, *args, **kwargs).arguments
        if trace or call.get("dummy_run", False):
            return original(self, *args, **kwargs)
        batch = call["input_batch"]
        if (
            batch.num_reqs != 1
            or batch.num_tokens != a
            or batch.input_ids[:a].tolist() != ids[:a]
            or batch.positions[:a].tolist() != list(range(a))
        ):
            raise ValueError(f"Native prefill does not match the saved prefix at {a}")
        boundaries = resolve_layers(self.speculative_config)
        if tuple(boundaries or ()) != tuple(i + 1 for i in layers):
            raise ValueError(f"Unexpected native capture boundaries: {boundaries}")
        aux = torch.cat([h[:a] for h in call["aux_hidden_states"]], -1).detach().cpu()
        if (
            aux.shape != reference.shape
            or aux.dtype != reference.dtype
            or not aux.isfinite().all()
        ):
            raise ValueError("Unexpected native aux shape, dtype or values")
        # Save the exact input only when the old capture cannot reproduce it.
        if not torch.equal(aux, reference):
            trace["aux"] = aux
        compute = self.model.compute_draft_logits
        sample = self._sample_logits

        def base(hidden):
            values = compute(hidden)
            trace["base"] = values[0].detach().cpu().clone()
            return values

        def final(logits, idx_map, sample_pos, step):
            if step == 0:
                prev = self.input_buffers.input_ids[self._anchor_idx[:1]].item()
                if sample_pos.tolist() != [a + 1] or prev != ids[a]:
                    raise ValueError(
                        f"Wrong native anchor ID/position at {a}: {prev}, {sample_pos}"
                    )
                trace["final"] = logits[0].detach().cpu().clone()
            return sample(logits, idx_map, sample_pos, step)

        with (
            patch.object(self.model, "compute_draft_logits", base),
            patch.object(self, "_sample_logits", final),
        ):
            return original(self, *args, **kwargs)

    if a < 1 or a + 1 >= len(ids):
        raise ValueError(f"Invalid saved anchor {a}")
    with patch.object(proposer, "propose", observe):
        engine.generate(
            [{"prompt_token_ids": ids[:a]}],
            params(
                temperature=1.0,
                max_tokens=2,
                ignore_eos=True,
                allowed_token_ids=[ids[a]],
                seed=42,
            ),
            use_tqdm=False,
        )
    for key in ("base", "final"):
        if (
            key not in trace
            or trace[key].shape != (vocab,)
            or not trace[key].isfinite().all()
        ):
            raise ValueError(
                f"Missing or invalid full-vocabulary native {key} logits at {a}"
            )
    return trace


def native_worker(out, name, export, pending, m):
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_USE_V2_MODEL_RUNNER="1")
    from vllm import LLM, SamplingParams, __version__
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
        get_eagle3_aux_layers_from_config,
    )

    cfg = m["drafts"][name]["config_values"]
    layers = m["aux_comparison"]["layers"]
    directory = out / "native-logits" / name
    runtime = {
        "torch": str(torch.__version__),
        "vllm": __version__,
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
    }
    check_identity(directory / "runtime.json", runtime)
    engine = LLM(
        model=m["target_dir"],
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=True,
        max_num_seqs=1,
        enable_prefix_caching=False,
        max_num_batched_tokens=m["max_length"] + 64,
        max_model_len=m["max_length"] + 64,
        speculative_config={
            "method": "dspark",
            "model": str(export),
            "attention_backend": "FLASH_ATTN",
            "draft_sample_method": "probabilistic",
            "num_speculative_tokens": cfg["block_size"],
            "enable_adaptive_verification": False,
        },
    )
    for s in pending:
        old = load_live(out, "vllm", s, layers, cfg["hidden_size"])["aux"]
        for a in s["anchors"]:
            path = trace_path(directory, s, a)
            if not path.exists():
                trace = capture_anchor(
                    engine,
                    DSparkSpeculator,
                    SamplingParams,
                    s["ids"],
                    a,
                    old[:a],
                    layers,
                    cfg["vocab_size"],
                    get_eagle3_aux_layers_from_config,
                )
                save(path, dict(row=s["row"], anchor=a, anchor_id=s["ids"][a], **trace))
        print(f"{name} native logits saved for row {s['row']}", flush=True)


def trace_path(directory, sample, anchor):
    return directory / f"row-{sample['row']}-a{anchor}.pt"


def differences(reference, native):
    delta = native.float() - reference.float()
    return {
        "rms": delta.square().mean().sqrt().item(),
        "max_abs": delta.abs().max().item(),
        "top1_match": float(reference.argmax() == native.argmax()),
    }


def anchor_scores(sf_base, sf_final, native_base, native_final, p):
    result = {}
    for head, sf, native in (
        ("base", sf_base, native_base),
        ("final", sf_final, native_final),
    ):
        if not sf.isfinite().all() or not native.isfinite().all():
            raise ValueError("Nonfinite draft logits")
        result[head] = differences(sf, native)
        q, r = sf.float().softmax(-1), native.float().softmax(-1)
        result[head].update(
            sf_overlap=torch.minimum(p, q).sum().item(),
            native_overlap=torch.minimum(p, r).sum().item(),
            distribution_tv=(q - r).abs().sum().item() / 2,
        )
    result["effective_bias"] = differences(
        sf_final.float() - sf_base.float(), native_final.float() - native_base.float()
    )
    return result


def replay(out, m, samples, name, wrapper):
    cfg = m["drafts"][name]["config_values"]
    layers = m["aux_comparison"]["layers"]
    directory = out / "native-logits" / name
    for s in samples:
        output = directory / f"row-{s['row']}.json"
        if output.exists():
            continue
        old = load_live(out, "vllm", s, layers, cfg["hidden_size"])["aux"]
        tr = load_live(out, "trainer", s, layers, cfg["hidden_size"])["final"]
        results = []
        for a in s["anchors"]:
            trace = torch.load(
                trace_path(directory, s, a), map_location="cpu", weights_only=True
            )
            if (trace["row"], trace["anchor"], trace["anchor_id"]) != (
                s["row"],
                a,
                s["ids"][a],
            ):
                raise ValueError("Native trace identity mismatch")
            aux = trace.get("aux", old[:a])
            # The SF block masks context at and after the anchor. Add the masked
            # row so token IDs and context have equal length in the training API.
            context = torch.cat([aux, torch.zeros_like(aux[:1])])[None].to(m["device"])
            ids = torch.tensor([s["ids"][: a + 1]], device=m["device"])
            pos = torch.tensor([[a]], device=m["device"])
            base, final, _, _ = predict(wrapper, ids, context, pos)
            target = wrapper.lm_head(tr[a].to(m["device"])).float().softmax(-1).cpu()
            item = anchor_scores(
                base[0, 0].cpu(),
                final[0, 0].cpu(),
                trace["base"],
                trace["final"],
                target,
            )
            item.update(
                anchor=a, anchor_id=s["ids"][a], aux_matches_stage3="aux" not in trace
            )
            results.append(item)
        save(output, {"row": s["row"], "anchors": results})


def report(out, samples):
    root = out / "native-logits"
    per_model = {}
    aggregates = {}
    for name in ("small", "large"):
        by_question = []
        for s in samples:
            r = read_json(root / name / f"row-{s['row']}.json")
            entries = r["anchors"]
            if r["row"] != s["row"] or [e["anchor"] for e in entries] != s["anchors"]:
                raise ValueError("Mismatched native replay rows or anchors")
            metrics = {
                f"{head}_{key}": sum(e[head][key] for e in entries) / len(entries)
                for head in ("base", "final", "effective_bias")
                for key in entries[0][head]
            }
            metrics["aux_matches_stage3"] = sum(
                e["aux_matches_stage3"] for e in entries
            ) / len(entries)
            by_question.append(metrics)
        per_model[name] = by_question
        aggregates[name] = {
            k: sum(q[k] for q in by_question) / len(by_question) for k in by_question[0]
        }

    picks = torch.randint(
        len(samples), (2000, len(samples)), generator=torch.Generator().manual_seed(42)
    )

    def estimate(x):
        ci = torch.quantile(x[picks].mean(1), x.new_tensor([0.025, 0.975]))
        return {
            "delta_pp": x.mean().item(),
            "ci95_pp": ci.tolist() if len(samples) > 1 else None,
        }

    comparisons = {}
    for head in ("base", "final"):
        diffs = {}
        for backend in ("sf", "native"):
            key = f"{head}_{backend}_overlap"
            diffs[backend] = 100 * torch.tensor(
                [
                    l[key] - s[key]
                    for s, l in zip(per_model["small"], per_model["large"], strict=True)
                ],
                dtype=torch.float64,
            )
        comparisons[head] = {k: estimate(v) for k, v in diffs.items()}
        comparisons[head]["native_minus_sf"] = estimate(diffs["native"] - diffs["sf"])
    summary = {
        "questions": len(samples),
        "anchors": sum(len(s["anchors"]) for s in samples),
        "models": aggregates,
        "large_minus_small": comparisons,
    }
    save(root / "summary.json", summary)
    for head, values in comparisons.items():
        print(
            f"{head} 首位接受率差（大表减小表）：SpecForge {values['sf']['delta_pp']:+.3f}，"
            f"vLLM 原生 {values['native']['delta_pp']:+.3f} 个百分点；"
            f"变化 {values['native_minus_sf']['delta_pp']:+.3f}，配对 95% 区间 {values['native_minus_sf']['ci95_pp']}"
        )
    for name, metrics in aggregates.items():
        print(
            f"{name} 同一 anchor 的 SpecForge/vLLM 对照："
            f"base top-1 一致率 {metrics['base_top1_match']:.2%}，"
            f"final top-1 一致率 {metrics['final_top1_match']:.2%}，"
            f"base/final 分布 TV {metrics['base_distribution_tv']:.5f}/{metrics['final_distribution_tv']:.5f}，"
            f"head 修正 RMS 差 {metrics['effective_bias_rms']:.5f}"
        )
    print(f"逐表 logits 误差与完整报告：{root/'summary.json'}")


def check_identity(path, identity):
    identity = json.loads(json.dumps(identity))
    if path.exists() and read_json(path) != identity:
        raise ValueError(
            f"{path}: native comparison inputs changed; use a separate native result directory"
        )
    save(path, identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--large-export", type=Path, required=True)
    parser.add_argument("--worker", choices=["small", "large"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.output.resolve()
    m = read_json(out / "manifest.json")
    samples = read_json(out / "prefixes.json")
    large = args.large_export.resolve()
    small = Path(m["aux_comparison"]["export"])
    exports = {"small": small, "large": large}
    if args.worker:
        directory = out / "native-logits" / args.worker
        pending = [
            s
            for s in samples
            if any(not trace_path(directory, s, a).exists() for a in s["anchors"])
        ]
        if pending:
            native_worker(out, args.worker, exports[args.worker], pending, m)
        return

    original = read_json(out / "summary.json")
    if (
        not original["complete"]
        or small == large
        or set(m["drafts"]) != {"small", "large"}
    ):
        raise ValueError(
            "Requires a completed small/large comparison and two distinct exports"
        )

    def verify(path, expected):
        if json.loads(json.dumps(fingerprint(path))) != expected:
            raise ValueError(f"Original comparison weights changed: {path}")

    verify(m["target_dir"], m["target"])
    verify(small, m["aux_comparison"]["weights"])
    for name, export in exports.items():
        s = m["drafts"][name]
        path = Path(s["path"])
        verify(draft_state(path) if draft_state(path).is_file() else path, s["weights"])
        if read_json(Path(s["config"])) != s["config_values"]:
            raise ValueError(f"Training config changed for {name}")
        eng = s["config_values"]["dflash_config"]["engram"]
        export_config = read_json(export / "config.json")
        if export_config.get("sample_from_anchor", True) != s["config_values"].get(
            "sample_from_anchor", True
        ):
            raise ValueError(f"{name}: export and training config sample_from_anchor differ")
        exported = export_config["dflash_config"]["engram"]
        if exported["bucket_counts"] != eng["bucket_counts"]:
            raise ValueError(
                f"{name}: export and checkpoint Engram bucket counts differ"
            )
    root = out / "native-logits"
    root.mkdir(exist_ok=True)
    invariant = m["aux_comparison"].get("batch_invariant")
    if invariant is None:
        os.environ.pop("VLLM_BATCH_INVARIANT", None)
    else:
        os.environ["VLLM_BATCH_INVARIANT"] = invariant
    check_identity(
        root / "manifest.json",
        {
            "original": file_id(out / "manifest.json"),
            "prefixes": file_id(out / "prefixes.json"),
            "large_export": fingerprint(large),
            "script": file_id(Path(__file__)),
            "replay_script": file_id(
                Path(__file__).with_name("diagnose_dspark_prefixes.py")
            ),
            "eager": True,
            "tp": 1,
            "batch_invariant": invariant,
        },
    )
    for name in exports:
        (root / name).mkdir(exist_ok=True)
        if any(
            not trace_path(root / name, s, a).exists()
            for s in samples
            for a in s["anchors"]
        ):
            subprocess.run(
                [
                    m["aux_comparison"]["vllm_python"],
                    __file__,
                    "--output",
                    str(out),
                    "--large-export",
                    str(large),
                    "--worker",
                    name,
                ],
                check=True,
            )
    from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
    from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead

    frozen = None
    for name, spec in m["drafts"].items():
        pending = [
            s for s in samples if not (root / name / f"row-{s['row']}.json").exists()
        ]
        if not pending:
            continue
        if frozen is None:
            frozen = TargetEmbeddingsAndHead.from_pretrained(
                m["target_dir"], device=m["device"]
            )
        model, _ = load_draft(Path(spec["config"]), Path(spec["path"]), m["device"])
        wrapper = OnlineDSparkModel(
            model,
            frozen.lm_head,
            frozen.embed_tokens,
            model.mask_token_id,
            block_size=model.block_size,
            attention_backend="sdpa",
        ).eval()
        with torch.inference_mode():
            replay(out, m, pending, name, wrapper)
        del wrapper, model
    report(out, samples)


if __name__ == "__main__":
    main()
