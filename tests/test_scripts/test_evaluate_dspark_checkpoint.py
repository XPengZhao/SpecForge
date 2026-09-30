from types import SimpleNamespace
from unittest import mock

import pytest

from scripts.evaluate_dspark_checkpoint import (
    checkpoint_step,
    evaluate_checkpoint,
    evaluation_config,
    write_tensorboard,
)
from specforge.config import Config


def make_config():
    return Config.model_validate(
        {
            "model": {
                "target_model_path": "target",
                "draft_model_config": "draft.json",
            },
            "data": {"hidden_states_path": "/unavailable-training-cache"},
            "training": {"strategy": "dspark", "num_anchors": 64},
        }
    )


def test_eval_config_uses_only_eval_cache_and_checkpoint_weights():
    cfg = make_config()
    cfg.training.resume_from = "/old-resume"
    cfg.training.resume_learning_rate = 1e-4
    cfg.training.resume_reset_data_position = True
    args = SimpleNamespace(
        config="train.yaml",
        overrides=[],
        checkpoint="/checkpoints/memory-step2604",
        eval_hidden_states_path="/gsm8k-eval",
        output="/results/memory2604.json",
    )
    with mock.patch("scripts.evaluate_dspark_checkpoint.load_config", return_value=cfg):
        result = evaluation_config(args)
    assert result.model.draft_checkpoint_path == args.checkpoint
    assert (
        result.data.hidden_states_path
        == result.data.eval_hidden_states_path
        == "/gsm8k-eval"
    )
    assert result.training.resume_from is None
    assert result.training.resume_learning_rate is None
    assert not result.training.resume_reset_data_position
    assert result.training.num_anchors == 64
    assert result.training.save_interval == 0
    assert result.tracking.report_to == "none"


@pytest.mark.parametrize("fail", [False, True])
def test_eval_never_trains_or_saves_and_closes_resources(fail):
    trainer = mock.Mock()
    trainer.evaluate.return_value = {"eval/tau_probabilistic": 4.5}
    if fail:
        trainer.evaluate.side_effect = RuntimeError("evaluation failed")
    run = mock.Mock(trainer=trainer)
    with mock.patch("specforge.training.assembly.build_training_run", return_value=run):
        if fail:
            with pytest.raises(RuntimeError, match="evaluation failed"):
                evaluate_checkpoint(make_config())
        else:
            assert evaluate_checkpoint(make_config()) == {"eval/tau_probabilistic": 4.5}
    run.run.assert_not_called()
    trainer.fit.assert_not_called()
    trainer.save_checkpoint.assert_not_called()
    trainer.evaluate.assert_called_once_with()
    trainer._loader.close.assert_called_once_with()
    trainer._logger.close.assert_called_once_with()
    trainer._controller.close_profiler.assert_called_once_with()


def test_tensorboard_step_resolves_checkpoint_and_latest_link(tmp_path):
    checkpoint = tmp_path / "memory-step2604"
    checkpoint.mkdir()
    latest = tmp_path / "memory-latest"
    latest.symlink_to(checkpoint, target_is_directory=True)
    assert checkpoint_step(str(checkpoint)) == 2604
    assert checkpoint_step(str(checkpoint / "training_state.pt")) == 2604
    assert checkpoint_step(str(latest)) == 2604
    assert checkpoint_step(str(tmp_path / "export"), 7812) == 7812
    with pytest.raises(ValueError, match="provide --step"):
        checkpoint_step(str(tmp_path / "export"))
    with pytest.raises(ValueError, match="non-negative"):
        checkpoint_step(str(checkpoint), -1)


def test_tensorboard_events_keep_eval_tags_and_checkpoint_steps(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    log_dir = str(tmp_path / "events")
    write_tensorboard(
        {"eval/tau_probabilistic": 4.5, "eval/per_position_acc": [0.8, 0.6]},
        log_dir,
        2604,
    )
    write_tensorboard({"eval/tau_probabilistic": 4.7}, log_dir, 3255)
    events = EventAccumulator(log_dir).Reload()
    points = events.Scalars("eval/tau_probabilistic")
    assert [point.step for point in points] == [2604, 3255]
    assert [point.value for point in points] == pytest.approx([4.5, 4.7])
    assert events.Scalars("eval/per_position_acc/0")[0].value == pytest.approx(0.8)
    assert not any(tag.startswith("train/") for tag in events.Tags()["scalars"])
