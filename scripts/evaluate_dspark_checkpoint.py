"""Evaluate an offline DSpark checkpoint without any optimizer updates.

Run from the repository root with PYTHONPATH=. and torchrun for multiple GPUs.
The configured draft architecture must match the checkpoint (including memory).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from contextlib import ExitStack
from pathlib import Path

from specforge.config import Config, load_config


def checkpoint_step(checkpoint: str, explicit_step: int | None = None) -> int:
    if explicit_step is not None:
        if explicit_step < 0:
            raise ValueError("--step must be non-negative")
        return explicit_step
    path = Path(checkpoint.removeprefix("file://")).resolve()
    if path.name == "training_state.pt":
        path = path.parent
    match = re.search(r"-step(\d+)$", path.name)
    if match is None:
        raise ValueError("Cannot infer checkpoint step from its path; provide --step")
    return int(match.group(1))


def write_tensorboard(metrics: dict, log_dir: str, step: int) -> None:
    from torch.utils.tensorboard import SummaryWriter

    from specforge.training.tracking import scalar_metrics

    with SummaryWriter(log_dir=log_dir) as writer:
        for name, value in scalar_metrics(metrics).items():
            writer.add_scalar(name, value, global_step=step)
        writer.flush()


def evaluation_config(args) -> Config:
    cfg = load_config(args.config, args.overrides)
    if cfg.training.strategy != "dspark" or cfg.mode != "offline":
        raise ValueError("This entry point requires an offline DSpark config")
    if cfg.deployment.mode != "local_colocated":
        raise ValueError("Evaluation requires deployment.mode=local_colocated")
    raw = cfg.model_dump()
    raw["model"]["draft_checkpoint_path"] = args.checkpoint
    # Assembly uses the ordinary offline runtime, but never traverses a training
    # loader. Point both sources at eval so the training cache is not required.
    raw["data"]["hidden_states_path"] = args.eval_hidden_states_path
    raw["data"]["eval_hidden_states_path"] = args.eval_hidden_states_path
    raw["data"]["offline_shuffle"] = False
    raw["training"].update(
        resume_from=None,
        resume_learning_rate=None,
        resume_reset_data_position=False,
        eval_interval=1,
        save_interval=0,
        role="all",
    )
    raw["profiling"]["enabled"] = False
    raw["tracking"]["report_to"] = "none"
    raw["run_id"] = "eval-" + Path(args.checkpoint).name
    raw["output_dir"] = str(Path(args.output).parent)
    return Config.model_validate(raw)


def evaluate_checkpoint(cfg: Config) -> dict:
    from specforge.application import resolve_run
    from specforge.training.assembly import build_training_run

    resolved = resolve_run(cfg)
    run = build_training_run(resolved.config, algorithm=resolved.algorithm)
    trainer = run.trainer
    # Do not call TrainingRun.run/Trainer.fit: both enter training and can write
    # checkpoints. The ordinary Evaluator handles eval mode and no_grad itself.
    with ExitStack() as cleanup:
        close_logger = getattr(trainer._logger, "close", None)
        if callable(close_logger):
            cleanup.callback(close_logger)
        cleanup.callback(trainer._controller.close_profiler)
        close_loader = getattr(trainer._loader, "close", None)
        if callable(close_loader):
            cleanup.callback(close_loader)
        metrics = trainer.evaluate()
    if not metrics:
        raise ValueError("Evaluation produced no batches; check the cache")
    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-hidden-states-path", required=True)
    parser.add_argument("--output", required=True, help="New JSON result file")
    parser.add_argument(
        "--tensorboard-dir",
        help="Write eval metrics here; reuse the directory across checkpoints of a run",
    )
    parser.add_argument(
        "--step",
        type=int,
        help="TensorBoard step; defaults to the checkpoint's -stepN suffix",
    )
    parser.add_argument("overrides", nargs="*", help="Optional dotted config overrides")
    args = parser.parse_args(argv)
    cfg = evaluation_config(args)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Result already exists: {output}; choose another output")
    step = None
    if args.tensorboard_dir or args.step is not None:
        step = checkpoint_step(args.checkpoint, args.step)
    if args.tensorboard_dir:
        # Fail before expensive model loading if the optional dependency is missing.
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401

    import torch.distributed as dist
    from accelerate.utils import set_seed

    from specforge.cli import _bootstrap_single_process_env
    from specforge.distributed import destroy_distributed, init_distributed

    os.environ["FSDP_SHARDING"] = cfg.training.fsdp_sharding
    _bootstrap_single_process_env()
    cfg.validate_world_size(int(os.environ["WORLD_SIZE"]))
    set_seed(cfg.training.seed)
    init_distributed(
        timeout=cfg.training.dist_timeout,
        tp_size=cfg.training.tp_size,
        sp_ulysses_size=cfg.training.sp_ulysses_size,
        sp_ring_size=cfg.training.sp_ring_size,
    )
    try:
        metrics = evaluate_checkpoint(cfg)
        if dist.get_rank() == 0:
            report = {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "eval_cache": str(Path(args.eval_hidden_states_path).resolve()),
                "config": cfg.model_dump(mode="json"),
                "metrics": metrics,
                "step": step,
                "tensorboard_dir": args.tensorboard_dir,
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
            if args.tensorboard_dir:
                write_tensorboard(metrics, args.tensorboard_dir, step)
                print(
                    f"TensorBoard saved to {args.tensorboard_dir}, step={step}",
                    flush=True,
                )
            print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
            print(f"Evaluation saved to {output}", flush=True)
    finally:
        destroy_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
