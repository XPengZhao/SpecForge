# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""On-disk checkpoint lifecycle: layout, rotation, best tracking, resume reads.

A checkpoint is ``{run_id}-step{N}/`` under ``output_dir``: ``training_state.pt``
(rank0 shared payload) plus ``training_state_rank{r}.pt`` per rank. FSDP optimizer
and all RNG state are rank-local; replicated DDP optimizer state is stored once
in the shared payload. Run-scoped ``{run_id}-latest``/``{run_id}-best`` links and
``{run_id}.best_meta.json`` sit beside them; multi-rank runs need a shared fs.
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
import re
import shutil
import tempfile
import time
import traceback
import uuid
from typing import Any, Dict, Iterator, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

STATE_FILE = "training_state.pt"
EVAL_META_FILE = "eval_meta.json"


def _cpu_tensors(obj: Any) -> Any:
    """Recursively move all tensors in *obj* to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    if isinstance(obj, dict):
        return {key: _cpu_tensors(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cpu_tensors(item) for item in obj)
    return obj


class CheckpointManager:
    def __init__(
        self,
        output_dir: str,
        run_id: str,
        *,
        max_checkpoints: int = 0,
        overwrite_checkpoints: bool = False,
        best_metric: str = "eval/simulated_acc_len",
        best_min_delta: float = 0.0,
    ) -> None:
        self.output_dir = output_dir
        self.run_id = run_id
        self.max_checkpoints = max_checkpoints
        self.overwrite_checkpoints = overwrite_checkpoints
        self.best_metric = best_metric
        self.best_min_delta = best_min_delta
        self.best_score: Optional[float] = None
        self.best_step: Optional[int] = None
        self._load_best_meta()

    # -- layout ------------------------------------------------------------
    def checkpoint_dir(self, step: int) -> str:
        return os.path.join(self.output_dir, f"{self.run_id}-step{step}")

    def _state_path(self, ckpt_dir: str) -> str:
        return os.path.join(ckpt_dir, STATE_FILE)

    @staticmethod
    def _rank_file(rank: int) -> str:
        return f"training_state_rank{rank}.pt"

    @property
    def _best_meta_path(self) -> str:
        return os.path.join(self.output_dir, f"{self.run_id}.best_meta.json")

    # -- save --------------------------------------------------------------
    def save(
        self,
        state: Optional[Dict[str, Any]],
        step: int,
        *,
        rank_state: Optional[Dict[str, Any]] = None,
        eval_metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Stage a complete checkpoint, replacing only this step if opted in.

        Backfills never invalidate later steps. Each save records evaluation
        metadata bound to this version of the weights, including an unscored
        marker when no evaluation ran. Collective: failures reach every rank.
        """
        ckpt_dir = self.checkpoint_dir(step)
        rank = self._rank()
        t0 = time.monotonic()
        _pfx = f"[ckpt step={step} rank={rank}]"

        err = ""
        staging_dir = None
        try:
            if self.is_rank0():
                if os.path.lexists(ckpt_dir):
                    if not self.overwrite_checkpoints:
                        raise FileExistsError(
                            f"checkpoint already exists: {ckpt_dir}; set "
                            "training.overwrite_checkpoints=true to replace only this step"
                        )
                    if os.path.islink(ckpt_dir) or not os.path.isdir(ckpt_dir):
                        raise ValueError(f"refusing to replace non-directory checkpoint: {ckpt_dir}")
                if state is None:
                    raise ValueError("rank0 must supply the shared checkpoint state")
                os.makedirs(self.output_dir, exist_ok=True)
                staging_dir = tempfile.mkdtemp(
                    prefix=f".{os.path.basename(ckpt_dir)}.pending-", dir=self.output_dir
                )
        except Exception as exc:
            err = f"prepare failed: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        self._all_ok(err)
        if torch.distributed.is_initialized():
            box = [staging_dir]
            torch.distributed.broadcast_object_list(box, src=0)
            staging_dir = box[0]

        try:
            if rank_state is not None:
                self._atomic_save(
                    rank_state,
                    os.path.join(staging_dir, self._rank_file(rank)),
                )
            print(f"{_pfx} rank_save done  (+{time.monotonic() - t0:.1f}s)\n", end="", flush=True)

            if self.is_rank0():
                self._atomic_save(state, self._state_path(staging_dir))
                self._atomic_json(
                    self._evaluation_meta(step, eval_metrics or {}, checkpoint_id=uuid.uuid4().hex),
                    os.path.join(staging_dir, EVAL_META_FILE),
                )
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

        try:
            self._all_ok(err)
        except RuntimeError:
            if self.is_rank0():
                shutil.rmtree(staging_dir)
            raise
        err = ""
        try:
            if self.is_rank0():
                self._commit_staged(staging_dir, ckpt_dir)
                self._load_best_meta()
                self._publish_best(self._best_record)
                self._rotate(keep_step=step)
                latest = self.latest_dir()
                if latest is not None:
                    self._point(f"{self.run_id}-latest", latest)
        except Exception as exc:
            err = f"commit failed: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        self._all_ok(err)
        if torch.distributed.is_initialized():
            box = [self.best_step, self.best_score]
            torch.distributed.broadcast_object_list(box, src=0)
            self.best_step, self.best_score = box
        print(f"{_pfx} rotate done  (+{time.monotonic() - t0:.1f}s)\n", end="", flush=True)

        self._barrier()  # no rank proceeds before the checkpoint is complete
        print(f"{_pfx} barrier done  (+{time.monotonic() - t0:.1f}s)\n", end="", flush=True)

        return ckpt_dir

    def _commit_staged(self, staging_dir: str, ckpt_dir: str) -> None:
        """Keep the old directory until every new rank shard has been written."""
        backup = staging_dir + ".previous"
        replacing = os.path.lexists(ckpt_dir)
        if replacing:
            if not self.overwrite_checkpoints:
                raise FileExistsError(f"checkpoint appeared during save: {ckpt_dir}")
            if os.path.islink(ckpt_dir) or not os.path.isdir(ckpt_dir):
                raise ValueError(f"refusing to replace non-directory checkpoint: {ckpt_dir}")
            os.replace(ckpt_dir, backup)
        try:
            os.replace(staging_dir, ckpt_dir)
        except Exception:
            if replacing:
                os.replace(backup, ckpt_dir)
            raise
        if replacing:
            try:
                shutil.rmtree(backup)
            except OSError as exc:
                logger.warning("saved checkpoint but could not remove old backup %s: %s", backup, exc)

    def _all_ok(self, err: str) -> None:
        # Barrier replacement: every rank learns every rank's outcome, so one
        # rank's FS failure raises everywhere instead of stranding the group.
        if torch.distributed.is_initialized():
            errs = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(errs, err)
        else:
            errs = [err]
        failures = [(r, e) for r, e in enumerate(errs) if e]
        if failures:
            raise RuntimeError(
                "checkpoint save failed: "
                + "; ".join(f"rank {r}: {e}" for r, e in failures)
            )

    # -- best tracking -----------------------------------------------------
    def score(self, eval_metrics: Optional[Dict[str, Any]]) -> Optional[float]:
        """Best-tracking score from an eval-metrics dict; None when absent."""
        s = (eval_metrics or {}).get(self.best_metric)
        s = float(s) if s is not None else None
        return s if s is not None and math.isfinite(s) else None

    def is_better(self, eval_metrics: Optional[Dict[str, Any]]) -> bool:
        """Rank-agreed verdict: do these metrics beat the tracked best (higher
        wins, by more than ``best_min_delta``)? False for empty metrics.
        Collective: rank0 decides and broadcasts, so every rank must call it.
        """
        verdict = False
        if self.is_rank0():
            s = self.score(eval_metrics)
            verdict = s is not None and (
                self.best_score is None or s > self.best_score + self.best_min_delta
            )
        return self._bcast_flag(verdict)

    def update_best(self, step: int, eval_metrics: Dict[str, Any]) -> None:
        """Record ``step`` as the tracked best unconditionally (callers gate with
        ``is_better``); rank0 repoints ``{run_id}-best`` and persists the meta."""
        self.best_score = self.score(eval_metrics)
        self.best_step = step
        if not self.is_rank0():
            return
        if not os.path.isfile(self._state_path(self.checkpoint_dir(step))):
            raise ValueError("best must reference a saved checkpoint")
        path = os.path.join(self.checkpoint_dir(step), EVAL_META_FILE)
        previous = self._read_json(path) or {}
        meta = self._evaluation_meta(
            step, eval_metrics, checkpoint_id=previous.get("checkpoint_id") or uuid.uuid4().hex
        )
        self._atomic_json(meta, path)
        self._publish_best(meta)

    def _load_best_meta(self) -> None:
        """Reconcile known scores; legacy unscored checkpoints are not guessed."""
        current = self._read_json(self._best_meta_path)
        best = current if self._valid_eval_meta(current) else None
        if current is not None and best is None:
            logger.warning("ignoring stale or incompatible best meta %s", self._best_meta_path)
        for step, path in sorted(self._all_checkpoints()):
            meta = self._read_json(os.path.join(path, EVAL_META_FILE))
            if not self._valid_eval_meta(meta, expected_step=step):
                continue
            if best is None or meta["score"] > best["score"] + self.best_min_delta:
                best = meta
        self._best_record = best
        self.best_score = best["score"] if best else None
        self.best_step = best["step"] if best else None

    @staticmethod
    def _read_json(path: str) -> Optional[dict]:
        try:
            with open(path) as fh:
                meta = json.load(fh)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            logger.warning("ignoring unreadable checkpoint meta %s: %s", path, exc)
            return None
        return meta if isinstance(meta, dict) else None

    def _valid_eval_meta(self, meta: Optional[dict], *, expected_step=None) -> bool:
        if meta is None:
            return False
        step = meta.get("step")
        score = meta.get("score")
        if (
            meta.get("run_id") != self.run_id
            or meta.get("metric") != self.best_metric
            or not isinstance(step, int)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or (expected_step is not None and step != expected_step)
            or not os.path.isfile(self._state_path(self.checkpoint_dir(step)))
        ):
            return False
        path = os.path.join(self.checkpoint_dir(step), EVAL_META_FILE)
        if os.path.exists(path):
            own = self._read_json(path)
            # An old global score must not attach to replacement weights.
            if own is None or any(
                own.get(key) != meta.get(key)
                for key in ("checkpoint_id", "run_id", "step", "metric", "score")
            ):
                return False
        return True

    def _evaluation_meta(self, step, metrics, *, checkpoint_id):
        return {
            "run_id": self.run_id, "step": step, "score": self.score(metrics),
            "metric": self.best_metric, "metrics": metrics,
            "checkpoint_id": checkpoint_id,
        }

    def _publish_best(self, meta: Optional[dict]) -> None:
        self._best_record = meta
        self.best_step = meta["step"] if meta else None
        self.best_score = meta["score"] if meta else None
        if meta is None:
            if os.path.isfile(self._best_meta_path):
                os.remove(self._best_meta_path)
            best_link = os.path.join(self.output_dir, f"{self.run_id}-best")
            if os.path.islink(best_link):
                os.unlink(best_link)
            return
        self._atomic_json(meta, self._best_meta_path)
        self._point(f"{self.run_id}-best", self.checkpoint_dir(meta["step"]))

    # -- load --------------------------------------------------------------
    def load(self, step: Optional[int] = None, *, map_location="cpu") -> Dict[str, Any]:
        """Load a checkpoint's shared state dict (``latest`` when ``step`` is None)."""
        target = self.checkpoint_dir(step) if step is not None else self.latest_dir()
        if target is None:
            raise FileNotFoundError(f"no checkpoint to load under {self.output_dir}")
        return torch.load(
            self._state_path(target), map_location=map_location, weights_only=False
        )

    @classmethod
    def read_resume_state(
        cls,
        path: str,
        *,
        map_location="cpu",
        require_full_state: bool = True,
    ) -> Dict[str, Any]:
        """Read a checkpoint into this rank's resume dict: the shared payload plus
        ``'backend'`` = this rank's ``training_state_rank{r}.pt`` content, passed
        through untouched (``{}`` when absent and ``require_full_state`` is False).
        Accepts a checkpoint dir, its ``training_state.pt``, a ``file://`` URI,
        or an output/run root containing exactly one checkpoint lineage.
        """
        path = cls.resolve_resume_dir(path)
        state = torch.load(
            os.path.join(path, STATE_FILE),
            map_location=map_location,
            weights_only=False,
        )
        initialized = torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if initialized else 0
        world = torch.distributed.get_world_size() if initialized else 1
        saved_world = state.get("world_size")
        rank_path = os.path.join(path, cls._rank_file(rank))
        if os.path.exists(rank_path) and (saved_world is None or saved_world == world):
            state["backend"] = torch.load(
                rank_path, map_location=map_location, weights_only=False
            )
            if (
                state["backend"].get("optimizer") is None
                and "replicated_optimizer_state" in state
            ):
                state["backend"]["optimizer"] = state["replicated_optimizer_state"]
            return state
        state["backend"] = {}
        if require_full_state and int(state.get("global_step") or 0) > 0:
            raise ValueError(
                f"checkpoint {path} has no usable per-rank state for rank {rank} at "
                f"world_size={world} (written at world_size={saved_world}): resuming "
                f"training needs each rank's optimizer/RNG shard — resume at the "
                f"original world size, or pass require_full_state=False to restore "
                f"weights and counters only"
            )
        return state

    @classmethod
    def resolve_resume_dir(cls, path: str) -> str:
        """Resolve one unambiguous resume target to its checkpoint directory.

        A run output directory is convenient in declarative configs, but it
        must never choose between unrelated runs silently.  Prefer its single
        complete ``*-latest`` pointer; when symlinks are unavailable, fall back
        to the highest complete ``<run-id>-stepN`` directory only if all
        candidates belong to the same run id.
        """
        path = str(path)
        if path.startswith("file://"):
            path = path[len("file://") :]
        path = os.path.abspath(os.path.expanduser(path))
        if os.path.basename(path) == STATE_FILE:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"checkpoint state file does not exist: {path}")
            path = os.path.dirname(path)
        if os.path.isfile(os.path.join(path, STATE_FILE)):
            return path
        if not os.path.isdir(path):
            raise FileNotFoundError(f"checkpoint path does not exist: {path}")

        latest = []
        for candidate in glob.glob(os.path.join(glob.escape(path), "*-latest")):
            resolved = os.path.realpath(candidate)
            if os.path.isfile(os.path.join(resolved, STATE_FILE)):
                latest.append(resolved)
        latest = sorted(set(latest))
        if len(latest) == 1:
            # Older versions could leave the pointer at a lower backfill step.
            # An explicit checkpoint/symlink path above still selects that step.
            match = re.match(r"^(.+)-step(\d+)$", os.path.basename(latest[0]))
            if match:
                run_id = match.group(1)
                pattern = re.compile(rf"^{re.escape(run_id)}-step(\d+)$")
                candidates = []
                for candidate in glob.glob(os.path.join(glob.escape(path), f"{glob.escape(run_id)}-step*")):
                    step_match = pattern.match(os.path.basename(candidate))
                    if step_match and os.path.isfile(os.path.join(candidate, STATE_FILE)):
                        candidates.append((int(step_match.group(1)), candidate))
                if candidates:
                    return max(candidates)[1]
            return latest[0]
        if len(latest) > 1:
            raise ValueError(
                f"resume root {path} contains multiple complete *-latest runs: "
                f"{latest}; select one checkpoint explicitly"
            )

        by_run: Dict[str, list[Tuple[int, str]]] = {}
        pattern = re.compile(r"^(.+)-step(\d+)$")
        for candidate in glob.glob(os.path.join(glob.escape(path), "*-step*")):
            match = pattern.match(os.path.basename(candidate))
            if match and os.path.isfile(os.path.join(candidate, STATE_FILE)):
                by_run.setdefault(match.group(1), []).append(
                    (int(match.group(2)), candidate)
                )
        if not by_run:
            raise FileNotFoundError(f"no complete checkpoint found under {path}")
        if len(by_run) > 1:
            raise ValueError(
                f"resume root {path} contains multiple checkpoint runs: "
                f"{sorted(by_run)}; select one checkpoint explicitly"
            )
        candidates = next(iter(by_run.values()))
        return max(candidates, key=lambda item: item[0])[1]

    def latest_dir(self) -> Optional[str]:
        """Directory of the newest complete checkpoint, or None."""
        step = self._max_step_on_disk()
        return self.checkpoint_dir(step) if step is not None else None

    # -- internals ---------------------------------------------------------
    @staticmethod
    def is_rank0() -> bool:
        return (
            not torch.distributed.is_initialized()
        ) or torch.distributed.get_rank() == 0

    @staticmethod
    def _rank() -> int:
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    @staticmethod
    def _barrier() -> None:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    @staticmethod
    def _bcast_flag(flag: bool) -> bool:
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() == 1
        ):
            return bool(flag)
        box = [bool(flag)]
        torch.distributed.broadcast_object_list(box, src=0)
        return bool(box[0])

    @staticmethod
    def _atomic_save(obj: Any, path: str) -> None:
        tmp = path + ".tmp"
        torch.save(
            _cpu_tensors(obj), tmp, _use_new_zipfile_serialization=False
        )
        os.replace(tmp, path)

    @staticmethod
    def _atomic_json(obj: Dict[str, Any], path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=2)
        os.replace(tmp, path)

    def _point(self, name: str, ckpt_dir: str) -> None:
        link = os.path.join(self.output_dir, name)
        if os.path.lexists(link) and not os.path.islink(link):
            logger.warning("not replacing non-symlink checkpoint pointer %s", link)
            return
        tmp = link + "." + uuid.uuid4().hex + ".tmp"
        try:
            # Relative target survives relocating output_dir to another mount.
            os.symlink(os.path.basename(ckpt_dir), tmp)
            os.replace(tmp, link)
        except OSError:
            return  # no symlink support: step dirs + best meta stay authoritative
        finally:
            if os.path.islink(tmp):
                os.unlink(tmp)

    def _rotate(self, keep_step: int) -> None:
        if self.max_checkpoints <= 0:
            return
        dirs = sorted(self._all_checkpoints())
        for step, path in dirs[: -self.max_checkpoints]:
            if step == self.best_step or step == keep_step:
                continue
            try:
                shutil.rmtree(path)
            except OSError as exc:
                # Never raise between collectives; a leftover dir is harmless.
                logger.warning("rotation could not remove %s: %s", path, exc)

    def _step_dirs(self) -> Iterator[Tuple[int, str]]:
        # Pending saves and backups do not match this run-scoped layout.
        pat = re.compile(rf"^{re.escape(self.run_id)}-step(\d+)$")
        pattern = os.path.join(
            glob.escape(self.output_dir), f"{glob.escape(self.run_id)}-step*"
        )
        for path in glob.glob(pattern):
            m = pat.match(os.path.basename(path))
            if m and os.path.isdir(path) and not os.path.islink(path):
                yield int(m.group(1)), path

    def _all_checkpoints(self) -> Iterator[Tuple[int, str]]:
        # Only dirs holding STATE_FILE count: a truncated save is never a target.
        for step, path in self._step_dirs():
            if os.path.isfile(self._state_path(path)):
                yield step, path

    def _max_step_on_disk(self) -> Optional[int]:
        steps = [step for step, _ in self._all_checkpoints()]
        return max(steps) if steps else None


__all__ = ["CheckpointManager"]
