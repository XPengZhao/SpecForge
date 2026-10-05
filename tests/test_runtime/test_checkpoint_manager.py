# coding=utf-8
"""CheckpointManager gates: atomic/complete-dir scanning, safe backfills,
run-scoped pointers and best meta, is_better/update_best API, read_resume_state,
plus a 2-process gloo pass over save + per-rank resume. CPU-only.
"""

import inspect
import json
import os
import shutil
import socket
import tempfile
import unittest
from unittest.mock import patch
from datetime import timedelta

import torch

METRIC = "eval/simulated_acc_len"
LOGGER = "specforge.training.checkpoint"


def _mgr(out, run_id="run", **kw):
    from specforge.training.checkpoint import CheckpointManager

    return CheckpointManager(out, run_id, **kw)


def _state(step, **extra):
    return {"draft_state_dict": {"w": torch.zeros(1)}, "global_step": step, **extra}


def _steps(mgr):
    return sorted(step for step, _ in mgr._all_checkpoints())


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestLayoutAndAtomicity(unittest.TestCase):
    def test_incomplete_dir_is_invisible(self):
        from specforge.training.checkpoint import STATE_FILE

        out = tempfile.mkdtemp(prefix="ckpt_atomic_")
        mgr = _mgr(out)
        mgr.save(_state(1), 1)
        mgr.save(_state(2), 2)
        # a truncated save: step dir exists but STATE_FILE never landed
        os.makedirs(os.path.join(out, "run-step5"))
        self.assertEqual(_steps(mgr), [1, 2])
        os.remove(os.path.join(out, "run-latest"))
        self.assertEqual(
            os.path.realpath(mgr.latest_dir()), os.path.realpath(mgr.checkpoint_dir(2))
        )
        leftovers = [
            f for _, _, files in os.walk(out) for f in files if f.endswith(".tmp")
        ]
        self.assertEqual(leftovers, [], "atomic writes must not leave .tmp files")
        self.assertTrue(os.path.isfile(os.path.join(mgr.checkpoint_dir(2), STATE_FILE)))

    def test_latest_ignores_and_preserves_non_symlink_shadows(self):
        from specforge.training.checkpoint import STATE_FILE

        out = tempfile.mkdtemp(prefix="ckpt_shadow_")
        mgr = _mgr(out)
        mgr.save(_state(1), 1)
        link = os.path.join(out, "run-latest")
        # materialized shadow: a real dir (with a plausible STATE_FILE) at the
        # link path, e.g. from a copier that dereferenced the symlink
        os.remove(link)
        os.makedirs(link)
        shutil.copy(os.path.join(mgr.checkpoint_dir(1), STATE_FILE), link)
        got = mgr.latest_dir()
        self.assertEqual(os.path.realpath(got), os.path.realpath(mgr.checkpoint_dir(1)))
        self.assertNotEqual(os.path.realpath(got), os.path.realpath(link))

        with self.assertLogs(LOGGER, level="WARNING"):
            mgr.save(_state(2), 2)
        self.assertTrue(os.path.isdir(link))
        self.assertFalse(os.path.islink(link))
        self.assertEqual(
            os.path.realpath(mgr.latest_dir()), os.path.realpath(mgr.checkpoint_dir(2))
        )

        shutil.rmtree(link)
        with open(link, "w") as fh:
            fh.write("x")
        self.assertEqual(
            os.path.realpath(mgr.latest_dir()),
            os.path.realpath(mgr.checkpoint_dir(2)),
        )
        with self.assertLogs(LOGGER, level="WARNING"):
            mgr.save(_state(3), 3)
        self.assertFalse(os.path.islink(link))
        with open(link) as fh:
            self.assertEqual(fh.read(), "x")
        self.assertEqual(
            os.path.realpath(mgr.latest_dir()), os.path.realpath(mgr.checkpoint_dir(3))
        )


class TestBackfill(unittest.TestCase):
    def test_overwrite_preserves_future_steps_and_best(self):
        out = tempfile.mkdtemp(prefix="ckpt_rewind_")
        mgr = _mgr(out, overwrite_checkpoints=True)
        for s in (1, 2, 3):
            mgr.save(_state(s), s)
        mgr.update_best(3, {METRIC: 5.0})
        self.assertEqual(mgr.best_step, 3)

        mgr.save(_state(2, replacement=True), 2)

        self.assertTrue(os.path.exists(mgr.checkpoint_dir(3)))
        self.assertTrue(os.path.exists(mgr.checkpoint_dir(1)))
        self.assertEqual(_steps(mgr), [1, 2, 3])
        self.assertTrue(mgr.load(2)["replacement"])
        self.assertEqual(mgr.best_step, 3)
        self.assertEqual(mgr.best_score, 5.0)
        self.assertTrue(os.path.exists(os.path.join(out, "run.best_meta.json")))
        self.assertTrue(os.path.lexists(os.path.join(out, "run-best")))
        self.assertEqual(
            os.path.realpath(mgr.latest_dir()), os.path.realpath(mgr.checkpoint_dir(3))
        )
        self.assertEqual(_mgr(out).best_step, 3)

    def test_overwrite_spares_best_below_step(self):
        out = tempfile.mkdtemp(prefix="ckpt_rewind_lo_")
        mgr = _mgr(out, overwrite_checkpoints=True)
        mgr.save(_state(1), 1)
        mgr.save(_state(2), 2)
        mgr.update_best(1, {METRIC: 5.0})
        mgr.save(_state(2), 2)
        self.assertEqual(mgr.best_step, 1)
        self.assertEqual(mgr.best_score, 5.0)
        self.assertTrue(os.path.exists(os.path.join(out, "run.best_meta.json")))

    def test_existing_step_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as out:
            mgr = _mgr(out)
            mgr.save(_state(2, original=True), 2)
            with self.assertRaisesRegex(RuntimeError, "overwrite_checkpoints=true"):
                mgr.save(_state(2, original=False), 2)
            self.assertTrue(mgr.load(2)["original"])

    def test_missing_older_step_does_not_move_latest_backwards(self):
        with tempfile.TemporaryDirectory() as out:
            mgr = _mgr(out)
            mgr.save(_state(21483), 21483, eval_metrics={METRIC: 5.5})
            mgr.save(_state(10416), 10416, eval_metrics={METRIC: 5.0})
            self.assertEqual(_steps(mgr), [10416, 21483])
            self.assertEqual(mgr.load()["global_step"], 21483)
            self.assertEqual(mgr.best_step, 21483)
            self.assertEqual(
                os.path.realpath(os.path.join(out, "run-latest")),
                os.path.realpath(mgr.checkpoint_dir(21483)),
            )

    def test_failed_overwrite_keeps_old_weights_scores_and_pointers(self):
        with tempfile.TemporaryDirectory() as out:
            mgr = _mgr(out, overwrite_checkpoints=True)
            mgr.save(_state(2, original=True), 2, eval_metrics={METRIC: 5.0})
            with patch.object(mgr, "_atomic_save", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(RuntimeError, "disk full"):
                    mgr.save(_state(2, original=False), 2)
            self.assertTrue(mgr.load(2)["original"])
            self.assertEqual(_mgr(out).best_score, 5.0)
            self.assertEqual(mgr.load()["global_step"], 2)

    def test_commit_failure_restores_old_directory(self):
        with tempfile.TemporaryDirectory() as out:
            mgr = _mgr(out, overwrite_checkpoints=True)
            mgr.save(_state(2, original=True), 2, eval_metrics={METRIC: 5.0})
            real_replace = os.replace

            def fail_publish(src, dst):
                if ".pending-" in src and not src.endswith(".previous") and dst == mgr.checkpoint_dir(2):
                    raise OSError("publish failed")
                return real_replace(src, dst)

            with patch("specforge.training.checkpoint.os.replace", side_effect=fail_publish):
                with self.assertRaisesRegex(RuntimeError, "publish failed"):
                    mgr.save(_state(2, original=False), 2)
            self.assertTrue(mgr.load(2)["original"])
            self.assertEqual(_mgr(out).best_score, 5.0)

    def test_overwriting_best_reselects_from_scored_checkpoints(self):
        with tempfile.TemporaryDirectory() as out:
            mgr = _mgr(out, overwrite_checkpoints=True)
            mgr.save(_state(1), 1, eval_metrics={METRIC: 4.5})
            mgr.save(_state(2), 2, eval_metrics={METRIC: 5.0})
            mgr.save(_state(3), 3, eval_metrics={METRIC: 4.0})
            mgr.save(_state(2), 2, eval_metrics={METRIC: 3.0})
            self.assertEqual(mgr.best_step, 1)
            self.assertEqual(mgr.best_score, 4.5)
            self.assertEqual(_mgr(out).best_step, 1)
            self.assertEqual(_steps(mgr), [1, 2, 3])

    def test_overwriting_only_best_without_eval_clears_stale_score(self):
        with tempfile.TemporaryDirectory() as out:
            mgr = _mgr(out, overwrite_checkpoints=True)
            mgr.save(_state(1), 1, eval_metrics={METRIC: 5.0})
            mgr.save(_state(1), 1)
            self.assertIsNone(mgr.best_step)
            self.assertIsNone(_mgr(out).best_step)
            self.assertFalse(os.path.lexists(os.path.join(out, "run-best")))
            self.assertFalse(os.path.exists(os.path.join(out, "run.best_meta.json")))

    def test_legacy_best_is_preserved_but_unscored_fallback_is_not_guessed(self):
        with tempfile.TemporaryDirectory() as out:
            from specforge.training.checkpoint import EVAL_META_FILE

            mgr = _mgr(out, overwrite_checkpoints=True)
            mgr.save(_state(1), 1)
            mgr.save(_state(2), 2)
            for step in (1, 2):
                os.remove(os.path.join(mgr.checkpoint_dir(step), EVAL_META_FILE))
            with open(mgr._best_meta_path, "w") as fh:
                json.dump({"run_id": "run", "step": 2, "score": 5.0, "metric": METRIC}, fh)
            mgr = _mgr(out, overwrite_checkpoints=True)
            self.assertEqual(mgr.best_step, 2)
            mgr.save(_state(3), 3)
            self.assertEqual(mgr.best_step, 2)
            mgr.save(_state(2), 2)
            self.assertIsNone(mgr.best_step)

    def test_stale_global_score_is_rejected_after_replacement(self):
        with tempfile.TemporaryDirectory() as out:
            mgr = _mgr(out, overwrite_checkpoints=True)
            mgr.save(_state(1), 1, eval_metrics={METRIC: 5.0})
            with open(mgr._best_meta_path) as fh:
                old_best = json.load(fh)
            mgr.save(_state(1), 1)
            # Simulate a crash between replacing weights and refreshing best.
            with open(mgr._best_meta_path, "w") as fh:
                json.dump(old_best, fh)
            self.assertIsNone(_mgr(out).best_step)


class TestRunScoping(unittest.TestCase):
    def test_two_run_ids_do_not_interact(self):
        out = tempfile.mkdtemp(prefix="ckpt_runs_")
        a = _mgr(out, "alpha", max_checkpoints=2, overwrite_checkpoints=True)
        b = _mgr(out, "beta")
        a.save(_state(1), 1)
        a.save(_state(2), 2)
        a.save(_state(3), 3)  # rotation drops alpha-step1 only
        b.save(_state(5), 5)
        a.update_best(3, {METRIC: 5.0})

        self.assertEqual(_steps(a), [2, 3])
        self.assertEqual(_steps(b), [5])
        self.assertEqual(
            os.path.realpath(a.latest_dir()), os.path.realpath(a.checkpoint_dir(3))
        )
        self.assertEqual(
            os.path.realpath(b.latest_dir()), os.path.realpath(b.checkpoint_dir(5))
        )
        self.assertTrue(os.path.exists(os.path.join(out, "alpha.best_meta.json")))
        self.assertFalse(os.path.exists(os.path.join(out, "beta.best_meta.json")))
        self.assertIsNone(_mgr(out, "beta").best_step)
        self.assertEqual(_mgr(out, "alpha").best_step, 3)

        a.save(_state(2), 2)  # replacing alpha-step2 must not touch beta
        self.assertTrue(os.path.exists(b.checkpoint_dir(5)))
        self.assertEqual(_steps(b), [5])

    def test_glob_metachar_run_id(self):
        from specforge.training.checkpoint import STATE_FILE

        out = tempfile.mkdtemp(prefix="ckpt_glob_")
        # decoys that an UNescaped "run[ab]-step*" glob would match
        for decoy in ("runa-step9", "runb-step7"):
            os.makedirs(os.path.join(out, decoy))
            with open(os.path.join(out, decoy, STATE_FILE), "wb") as fh:
                fh.write(b"x")
        mgr = _mgr(out, "run[ab]")
        mgr.save(_state(1), 1)
        self.assertEqual(_steps(mgr), [1])
        os.remove(os.path.join(out, "run[ab]-latest"))
        self.assertEqual(
            os.path.realpath(mgr.latest_dir()), os.path.realpath(mgr.checkpoint_dir(1))
        )
        mgr.save(_state(0), 0)
        self.assertTrue(os.path.exists(mgr.checkpoint_dir(1)))
        self.assertTrue(os.path.exists(os.path.join(out, "runa-step9")))
        self.assertTrue(os.path.exists(os.path.join(out, "runb-step7")))


class TestBestMeta(unittest.TestCase):
    def _seeded(self):
        out = tempfile.mkdtemp(prefix="ckpt_meta_")
        _mgr(out).save(_state(1), 1)
        return out, os.path.join(out, "run.best_meta.json")

    def _assert_blind(self, out):
        with self.assertLogs(LOGGER, level="WARNING"):
            mgr = _mgr(out)
        self.assertIsNone(mgr.best_step)
        self.assertIsNone(mgr.best_score)

    def test_truncated_json_ignored(self):
        out, meta = self._seeded()
        with open(meta, "w") as fh:
            fh.write('{"run_id": "run", "step"')
        self._assert_blind(out)

    def test_wrong_run_id_ignored(self):
        out, meta = self._seeded()
        with open(meta, "w") as fh:
            json.dump(
                {"run_id": "other", "metric": METRIC, "step": 1, "score": 5.0}, fh
            )
        self._assert_blind(out)

    def test_wrong_metric_ignored(self):
        out, meta = self._seeded()
        with open(meta, "w") as fh:
            json.dump(
                {"run_id": "run", "metric": "eval/other", "step": 1, "score": 5.0}, fh
            )
        self._assert_blind(out)

    def test_missing_step_dir_ignored(self):
        out, meta = self._seeded()
        with open(meta, "w") as fh:
            json.dump({"run_id": "run", "metric": METRIC, "step": 42, "score": 5.0}, fh)
        self._assert_blind(out)

    def test_valid_meta_rehydrates(self):
        out, _ = self._seeded()
        _mgr(out).update_best(1, {METRIC: 5.0})
        mgr = _mgr(out)
        self.assertEqual(mgr.best_step, 1)
        self.assertEqual(mgr.best_score, 5.0)

    def test_rebuild_from_per_checkpoint_scores_without_global_meta(self):
        from specforge.training.checkpoint import EVAL_META_FILE

        with tempfile.TemporaryDirectory() as out:
            mgr = _mgr(out)
            mgr.save(_state(1), 1, eval_metrics={METRIC: 5.0})
            mgr.save(_state(2), 2, eval_metrics={METRIC: 4.0})
            self.assertTrue(os.path.isfile(os.path.join(mgr.checkpoint_dir(2), EVAL_META_FILE)))
            os.remove(mgr._best_meta_path)
            self.assertEqual(_mgr(out).best_step, 1)

    def test_nonfinite_score_is_not_best(self):
        with tempfile.TemporaryDirectory() as out:
            mgr = _mgr(out)
            for step, score in enumerate((float("nan"), float("inf"), -float("inf")), 1):
                self.assertFalse(mgr.is_better({METRIC: score}))
                mgr.save(_state(step), step, eval_metrics={METRIC: score})
            self.assertIsNone(mgr.best_step)


class TestBestApi(unittest.TestCase):
    def test_is_better_gates_and_min_delta(self):
        out = tempfile.mkdtemp(prefix="ckpt_best_")
        mgr = _mgr(out, best_min_delta=0.5)
        self.assertFalse(mgr.is_better(None))
        self.assertFalse(mgr.is_better({}))
        self.assertFalse(mgr.is_better({"eval/other": 1.0}))
        self.assertTrue(mgr.is_better({METRIC: 1.0}))

        mgr.save(_state(1), 1)
        self.assertIsNone(mgr.update_best(1, {METRIC: 1.0}))
        self.assertFalse(mgr.is_better({METRIC: 1.0}))
        self.assertFalse(mgr.is_better({METRIC: 1.5}))  # not > best + min_delta
        self.assertTrue(mgr.is_better({METRIC: 1.6}))

    def test_update_best_has_no_force_param(self):
        from specforge.training.checkpoint import CheckpointManager

        params = inspect.signature(CheckpointManager.update_best).parameters
        self.assertNotIn("force", params)

    def test_update_best_records_unconditionally(self):
        # caller gates with is_better; update_best itself never refuses
        out = tempfile.mkdtemp(prefix="ckpt_best_uncond_")
        mgr = _mgr(out)
        mgr.save(_state(1), 1)
        mgr.save(_state(2), 2)
        mgr.update_best(1, {METRIC: 5.0})
        mgr.update_best(2, {METRIC: 1.0})
        self.assertEqual(mgr.best_step, 2)
        self.assertEqual(mgr.best_score, 1.0)


class TestReadResumeState(unittest.TestCase):
    def test_root_resume_ignores_stale_lower_latest_pointer(self):
        from specforge.training.checkpoint import CheckpointManager

        with tempfile.TemporaryDirectory() as out:
            mgr = _mgr(out)
            mgr.save(_state(3), 3)
            mgr.save(_state(5), 5)
            mgr._point("run-latest", mgr.checkpoint_dir(3))
            self.assertEqual(CheckpointManager.read_resume_state(out, require_full_state=False)["global_step"], 5)
            self.assertEqual(
                CheckpointManager.read_resume_state(mgr.checkpoint_dir(3), require_full_state=False)["global_step"], 3
            )

    def test_replicated_optimizer_is_restored_from_shared_state(self):
        from specforge.training.checkpoint import CheckpointManager

        out = tempfile.mkdtemp(prefix="ckpt_replicated_")
        state = _state(
            3,
            world_size=1,
            replicated_optimizer_state={"moment": torch.tensor([7.0])},
        )
        ckpt = _mgr(out).save(
            state,
            3,
            rank_state={"optimizer": None, "rng": {"torch": torch.tensor([1])}},
        )

        loaded = CheckpointManager.read_resume_state(ckpt)

        self.assertEqual(loaded["backend"]["optimizer"]["moment"].item(), 7.0)
        self.assertEqual(loaded["backend"]["rng"]["torch"].item(), 1)

    def test_backend_passthrough_and_path_forms(self):
        from specforge.training.checkpoint import STATE_FILE, CheckpointManager

        out = tempfile.mkdtemp(prefix="ckpt_read_")
        rng = torch.get_rng_state()
        rank_state = {"optimizer": {"lr": 0.5}, "rng": {"torch": rng}, "custom": 7}
        ckpt = _mgr(out).save(_state(3, world_size=1), 3, rank_state=rank_state)

        for target in (ckpt, os.path.join(ckpt, STATE_FILE), f"file://{ckpt}"):
            st = CheckpointManager.read_resume_state(target)
            self.assertEqual(st["global_step"], 3)
            self.assertEqual(st["backend"]["optimizer"], {"lr": 0.5})
            self.assertEqual(st["backend"]["custom"], 7)
            self.assertTrue(torch.equal(st["backend"]["rng"]["torch"], rng))
            self.assertNotIn("optimizer_state_dict", st)
            self.assertNotIn("rng_state", st)

        root_state = CheckpointManager.read_resume_state(out)
        self.assertEqual(root_state["global_step"], 3)

    def test_run_root_without_symlinks_uses_latest_complete_step(self):
        from specforge.training.checkpoint import CheckpointManager

        out = tempfile.mkdtemp(prefix="ckpt_root_")
        mgr = _mgr(out)
        mgr.save(_state(2, world_size=1), 2, rank_state={"optimizer": {}})
        mgr.save(_state(7, world_size=1), 7, rank_state={"optimizer": {}})
        os.remove(os.path.join(out, "run-latest"))

        state = CheckpointManager.read_resume_state(out)
        self.assertEqual(state["global_step"], 7)

    def test_ambiguous_multi_run_root_is_rejected(self):
        from specforge.training.checkpoint import CheckpointManager

        out = tempfile.mkdtemp(prefix="ckpt_ambiguous_")
        _mgr(out, "alpha").save(
            _state(2, world_size=1), 2, rank_state={"optimizer": {}}
        )
        _mgr(out, "beta").save(_state(3, world_size=1), 3, rank_state={"optimizer": {}})
        with self.assertRaisesRegex(ValueError, "multiple complete.*latest"):
            CheckpointManager.read_resume_state(out)

    def test_require_full_state_raises_without_rank_file(self):
        from specforge.training.checkpoint import CheckpointManager

        out = tempfile.mkdtemp(prefix="ckpt_norank_")
        ckpt = _mgr(out).save(_state(5, world_size=1), 5)
        with self.assertRaisesRegex(ValueError, "per-rank state"):
            CheckpointManager.read_resume_state(ckpt)
        st = CheckpointManager.read_resume_state(ckpt, require_full_state=False)
        self.assertEqual(st["backend"], {})
        self.assertEqual(st["global_step"], 5)

    def test_step0_weights_only_is_fine(self):
        from specforge.training.checkpoint import CheckpointManager

        out = tempfile.mkdtemp(prefix="ckpt_step0_")
        ckpt = _mgr(out).save(_state(0, world_size=1), 0)
        st = CheckpointManager.read_resume_state(ckpt)  # require_full_state default
        self.assertEqual(st["backend"], {})

    def test_world_size_mismatch(self):
        from specforge.training.checkpoint import CheckpointManager

        out = tempfile.mkdtemp(prefix="ckpt_ws_")
        ckpt = _mgr(out).save(
            _state(4, world_size=4), 4, rank_state={"optimizer": {}, "rng": {}}
        )
        with self.assertRaisesRegex(ValueError, r"written at world_size=4"):
            CheckpointManager.read_resume_state(ckpt)
        st = CheckpointManager.read_resume_state(ckpt, require_full_state=False)
        self.assertEqual(st["backend"], {})


def _dist_worker(rank, world, port, out_dir, results_dir):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=60),
    )
    from specforge.training.checkpoint import CheckpointManager

    mgr = CheckpointManager(out_dir, "dist", overwrite_checkpoints=True)
    ckpt = mgr.save(
        {"global_step": 3, "world_size": world},
        3,
        rank_state={"optimizer": {"rank": rank}, "rng": {"torch": [rank]}},
    )
    verdict = mgr.is_better({METRIC: 2.0})  # collective: every rank must call
    if verdict:
        mgr.update_best(3, {METRIC: 2.0})
    # A rank-local failure during overwrite must reach all ranks and leave the
    # previous shared weights, shards, and best record untouched.
    original_atomic_save = mgr._atomic_save

    def fail_rank_save(obj, path):
        if rank == 1:
            raise OSError("rank1 disk full")
        return original_atomic_save(obj, path)

    failed = False
    with patch.object(mgr, "_atomic_save", side_effect=fail_rank_save):
        try:
            mgr.save(
                {"global_step": 3, "world_size": world, "replacement": True}, 3,
                rank_state={"optimizer": {"rank": rank}, "rng": {"torch": [99]}},
            )
        except RuntimeError as exc:
            failed = "rank1 disk full" in str(exc)
    mgr.save(
        {"global_step": 1, "world_size": world}, 1,
        rank_state={"optimizer": {"rank": rank}, "rng": {"torch": [rank]}},
        eval_metrics={METRIC: 3.0},
    )
    st = CheckpointManager.read_resume_state(ckpt)
    with open(os.path.join(results_dir, f"rank{rank}.json"), "w") as fh:
        json.dump(
            {
                "ckpt": ckpt,
                "files": sorted(os.listdir(ckpt)),
                "backend_rank": st["backend"]["optimizer"]["rank"],
                "backend_rng": st["backend"]["rng"]["torch"],
                "global_step": st["global_step"],
                "verdict": bool(verdict),
                "best_step": mgr.best_step,
                "best_score": mgr.best_score,
                "latest_step": mgr.load()["global_step"],
                "overwrite_failed": failed,
                "old_weights_preserved": not st.get("replacement", False),
            },
            fh,
        )
    dist.destroy_process_group()


class TestDistributedSaveAndResume(unittest.TestCase):
    def test_two_rank_save_and_per_rank_resume(self):
        import torch.multiprocessing as mp

        from specforge.training.checkpoint import EVAL_META_FILE, STATE_FILE

        out = tempfile.mkdtemp(prefix="ckpt_dist_")
        results = tempfile.mkdtemp(prefix="ckpt_dist_res_")
        mp.spawn(
            _dist_worker, args=(2, _free_port(), out, results), nprocs=2, join=True
        )

        loaded = []
        for r in range(2):
            with open(os.path.join(results, f"rank{r}.json")) as fh:
                loaded.append(json.load(fh))
        self.assertEqual(loaded[0]["ckpt"], loaded[1]["ckpt"])
        expected = sorted(
            [STATE_FILE, EVAL_META_FILE, "training_state_rank0.pt", "training_state_rank1.pt"]
        )
        for r, res in enumerate(loaded):
            self.assertEqual(res["files"], expected)
            self.assertEqual(res["backend_rank"], r)  # each rank got its own shard
            self.assertEqual(res["backend_rng"], [r])
            self.assertEqual(res["global_step"], 3)
            self.assertTrue(res["verdict"])
            self.assertEqual(res["best_step"], 1)
            self.assertEqual(res["best_score"], 3.0)
            self.assertEqual(res["latest_step"], 3)
            self.assertTrue(res["overwrite_failed"])
            self.assertTrue(res["old_weights_preserved"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
