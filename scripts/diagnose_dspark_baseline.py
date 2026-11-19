"""Compare a vanilla DSpark checkpoint with two completed trainer-aux replays."""

import argparse
import hashlib
from pathlib import Path
from statistics import mean

from diagnose_dspark_prefixes import (
    diagnose_question,
    draft_state,
    fingerprint,
    load_draft,
    load_live,
    read_json,
    save,
)
import torch


def run(out, config_path, checkpoint):
    manifest = read_json(out / "manifest.json")
    if not read_json(out / "summary.json")["complete"]:
        raise ValueError("Finish the original three-stage diagnosis first")
    layers = manifest["aux_comparison"]["layers"]
    target = Path(manifest["target_dir"])
    config = read_json(config_path)
    ref = manifest["drafts"]["small"]["config_values"]
    opts = config["dflash_config"]
    if (
        opts.get("markov_head_type") != "vanilla"
        or opts["target_layer_ids"] != layers
        or any(config[k] != ref[k] for k in ("block_size", "model_type", "vocab_size"))
        or config.get("draft_vocab_size") not in (None, ref["vocab_size"])
    ):
        raise ValueError(
            "Baseline must use vanilla DSpark and match the cached layers, block size and vocabulary"
        )
    if [list(x) for x in fingerprint(target)] != manifest["target"]:
        raise ValueError("Target weights/config changed since the original diagnosis")

    groups = {
        name: read_json(out / filename)
        for name, filename in (
            ("Open PerfectBlend", "perfectblend-prefixes.json"),
            ("GSM8K", "prefixes.json"),
        )
    }
    dest = out / "baseline-comparison"
    dest.mkdir(exist_ok=True)
    weights = draft_state(checkpoint)
    identity = {
        "config": config,
        "weights": [
            list(x) for x in fingerprint(weights if weights.is_file() else checkpoint)
        ],
        "run": hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest(),
        "prefixes": {
            name: hashlib.sha256((out / name).read_bytes()).hexdigest()
            for name in ("prefixes.json", "perfectblend-prefixes.json")
        },
    }
    stamp = dest / "inputs.json"
    if stamp.exists() and read_json(stamp) != identity:
        raise ValueError(f"Baseline inputs changed; move {dest} aside before rerunning")
    save(stamp, identity)
    device = manifest["device"]
    wrapper = None
    report = {}
    for dataset, samples in groups.items():
        rows = []
        for sample in samples:
            filename = f"row-{sample['row']}.json"
            path = dest / filename
            if not path.exists():
                if wrapper is None:
                    from specforge.algorithms.common.dflash_family_model import (
                        OnlineDSparkModel,
                    )
                    from specforge.modeling.target.target_utils import (
                        TargetEmbeddingsAndHead,
                    )

                    model, contract = load_draft(config_path, checkpoint, device)
                    expected_step = manifest.get("expected_step")
                    step = contract.get("global_step")
                    if (
                        expected_step is not None
                        and step is not None
                        and int(step) != expected_step
                    ):
                        raise ValueError(
                            f"Baseline step {step}, expected {expected_step}"
                        )
                    if step is None:
                        print(
                            f"Baseline step unavailable; verify checkpoint provenance (expected {expected_step})",
                            flush=True,
                        )
                    frozen = TargetEmbeddingsAndHead.from_pretrained(
                        str(target), device=device
                    )
                    wrapper = OnlineDSparkModel(
                        model,
                        frozen.lm_head,
                        frozen.embed_tokens,
                        model.mask_token_id,
                        block_size=model.block_size,
                        attention_backend="sdpa",
                    ).eval()
                    width = read_json(target / "config.json")["hidden_size"]
                data = load_live(out, "trainer", sample, layers, width)
                capture = {
                    "context": dict(
                        zip(layers, data["aux"].split(width, -1), strict=True)
                    ),
                    "final": data["final"],
                }
                diagnose_question(wrapper, capture, sample, path, False, device)
            trio = [
                read_json(p)
                for p in (path, out / "small" / filename, out / "large" / filename)
            ]
            if any(
                r["anchors"] != sample["anchors"] or r["row"] != sample["row"]
                for r in trio
            ):
                raise ValueError(f"Unpaired rows for {dataset} {filename}")
            rows.append(trio)
        report[dataset] = {}
        for metric in ("base_overlap", "head_overlap_gain", "final_overlap"):
            print(f"\n{dataset} ({len(rows)} questions), {metric}, trainer aux")
            print(
                "pos   baseline     0.13M      1.3M    small-base   large-base  large-small (pp)"
            )
            positions = []
            for pos in range(config["block_size"]):
                b, s, l = [
                    100 * mean(row[m]["per_position"][metric][pos] for row in rows)
                    for m in range(3)
                ]
                print(
                    f"{pos:3} {b:10.3f} {s:9.3f} {l:9.3f} {s-b:+12.3f} {l-b:+12.3f} {l-s:+12.3f}"
                )
                positions.append(
                    {
                        "pos": pos,
                        "baseline": b,
                        "small": s,
                        "large": l,
                        "small_minus_baseline_pp": s - b,
                        "large_minus_baseline_pp": l - b,
                        "large_minus_small_pp": l - s,
                    }
                )
            report[dataset][metric] = positions
    save(dest / "summary.json", report)
    print(f"Report: {dest / 'summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("output", "config", "checkpoint"):
        parser.add_argument("--" + arg, required=True, type=Path)
    args = parser.parse_args()
    with torch.inference_mode():
        run(args.output, args.config, args.checkpoint)
