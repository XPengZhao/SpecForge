"""Managed stream reconstruction: cursor, launch roles and actual capture order."""
import json
from copy import deepcopy
from unittest import mock

import pytest

from specforge.training.online_resume import PLAN_KEY, prompt_plan, resume_cursor, validate_resume_plan
from specforge.launch import _epoch_online_prompts


def plan_and_state():
    prompts = [{"task_id": f"p{i}", "payload": {"input_ids": [i]}} for i in range(5)]
    plan = prompt_plan(prompts, seed=42, epochs=3, dp_size=2, batch_size=1, accumulation_steps=2)
    state = dict(global_step=2, epoch=0, epoch_batch=4, epoch_samples=4,
                 world_size=2, batch_size=1, accumulation_steps=2, **{PLAN_KEY: plan})
    return prompts, plan, state


def test_cursor_and_fingerprint_fail_closed():
    prompts, plan, state = plan_and_state()
    assert validate_resume_plan(state, plan, prompts) == 8
    for key in ('seed', 'epochs', 'sha256', 'num_prompts'):
        changed = dict(plan, **{key: 'changed'})
        with pytest.raises((ValueError, TypeError)):
            validate_resume_plan(state, changed, prompts)
    for key in ('epoch', 'epoch_batch', 'epoch_samples', 'world_size', 'batch_size', 'accumulation_steps'):
        with pytest.raises(ValueError):
            resume_cursor(dict(state, **{key: 99}), plan)
    changed_prompts = deepcopy(prompts)
    changed_prompts[0]['payload']['input_ids'][0] += 1
    assert prompt_plan(changed_prompts, seed=42, epochs=3, dp_size=2, batch_size=1,
                       accumulation_steps=2)['sha256'] != plan['sha256']


def test_legacy_requires_and_verifies_original_prefix(tmp_path):
    prompts, plan, state = plan_and_state()
    del state[PLAN_KEY]
    with pytest.raises(ValueError, match='resume_ref_channel'):
        validate_resume_plan(state, plan, prompts)
    expected = [p for epoch in range(3) for p in _epoch_online_prompts(prompts, epoch, 3, seed=42)]
    path = tmp_path / 'refs.jsonl'
    lines = [json.dumps({'source_task_id': p['task_id']}) + '\n' for p in expected]
    path.write_text(''.join(lines))
    assert validate_resume_plan(state, plan, prompts, legacy_channel=path) == 8
    lines[1], lines[2] = lines[2], lines[1]
    path.write_text(''.join(lines))
    with pytest.raises(ValueError, match='order differs'):
        validate_resume_plan(state, plan, prompts, legacy_channel=path)
    path.write_text('')
    with pytest.raises(ValueError, match='ends before'):
        validate_resume_plan(state, plan, prompts, legacy_channel=path)


@pytest.mark.parametrize('skip', [0, 4, 5, 8, 15])
def test_rebuilt_producer_emits_exact_suffix_across_epochs(tmp_path, skip):
    from tests.test_runtime.test_disagg_multiserver import (
        _FakeMooncakeStore, _StubCaptureServer, MooncakeFeatureStore,
        StreamingRefChannel, _adapter, _build, _prompts, _published_refs,
    )
    backend = _FakeMooncakeStore()
    server = _StubCaptureServer(backend)
    store = MooncakeFeatureStore(store=backend, store_id='run0')
    channel = StreamingRefChannel(str(tmp_path / 'refs.jsonl'))
    prompts = _prompts(5)
    _, drive = _build([_adapter(store, server)], prompts, store, channel,
                     lease=2, prompt_epochs=3, prompt_seed=42,
                     prompt_skip=skip, ordered_prompts=True)
    assert drive() == 15 - skip
    expected = [p['task_id'] for epoch in range(3)
                for p in _epoch_online_prompts(prompts, epoch, 3, seed=42)]
    actual = [r.source_task_id for r in _published_refs(channel.path)]
    assert actual == expected[skip:]
    assert len(actual) == len(set(actual))


def test_managed_resume_launches_both_children_with_fresh_state(tmp_path):
    from tests.test_runtime.test_launch_plan import _managed_config, ALGORITHM, CAPTURE_CONTRACT
    from specforge.config import Config
    from specforge.launch_plan import build_launch_plan
    from specforge.cli import _config_for_role
    raw = _managed_config(str(tmp_path / 'new')).model_dump()
    raw['training']['resume_from'] = '/old/run-step1962'
    cfg = Config.model_validate(raw)
    with mock.patch('specforge.training.capture_contract.resolve_server_capture_contract',
                    return_value=CAPTURE_CONTRACT):
        plan = build_launch_plan(cfg, algorithm=ALGORITHM, config_path='run.yaml', env={})
    assert plan.kind == 'managed_supervisor'
    assert plan.role == 'both'
    producer, consumer = plan.commands
    (tmp_path / 'new' / 'logs').mkdir(parents=True)
    for role, child in zip(('producer', 'consumer'), (producer, consumer)):
        env = dict(child.env)
        if role == 'consumer':
            env.update(RANK='0', WORLD_SIZE='2', LOCAL_RANK='0',
                       MASTER_ADDR='127.0.0.1', MASTER_PORT='29500')
        child_plan = build_launch_plan(cfg, algorithm=ALGORITHM, config_path='run.yaml',
                                       requested_role=role, env=env)
        assert child_plan.kind == 'worker'
    for child in (producer, consumer):
        assert child.env['SPECFORGE_ONLINE_REBUILD_CHECKPOINT'] == '/old/run-step1962'
    with mock.patch.dict('os.environ', producer.env):
        child = _config_for_role(cfg, 'producer')
        assert child.training.resume_from is None
    with mock.patch.dict('os.environ', consumer.env):
        child = _config_for_role(cfg, 'consumer')
        assert child.training.resume_from == '/old/run-step1962'

@pytest.mark.parametrize('failure', ['capture', 'reorder', 'omitted'])
def test_ordered_capture_fails_instead_of_retrying_out_of_order(tmp_path, failure):
    from tests.test_runtime.test_disagg_multiserver import (
        _FakeMooncakeStore, _StubCaptureServer, MooncakeFeatureStore,
        StreamingRefChannel, _adapter, _build, _prompts, _published_refs,
    )
    from specforge.inference.rollout_worker import RolloutWorker
    backend = _FakeMooncakeStore()
    store = MooncakeFeatureStore(store=backend, store_id='run0')
    channel = StreamingRefChannel(str(tmp_path / 'refs.jsonl'))
    _, drive = _build([_adapter(store, _StubCaptureServer(backend))],
                     _prompts(3), store, channel, lease=3, ordered_prompts=True)
    original = RolloutWorker.run_once
    def broken(worker, **kwargs):
        if failure == 'capture':
            raise RuntimeError('capture unavailable')
        refs = original(worker, **kwargs)
        if failure == 'omitted':
            worker._recent_failures.append('server_capture: simulated store write failure')
            return refs[1:]
        return list(reversed(refs))
    with mock.patch.object(RolloutWorker, 'run_once', broken):
        with pytest.raises(RuntimeError, match='ordered capture failed|changed prompt order') as error:
            drive()
    if failure == 'omitted':
        assert 'server_capture: simulated store write failure' in str(error.value)
        assert 'expected_task_id=' in str(error.value)
        assert 'actual_task_id=' in str(error.value)
        assert 'returned_refs=2' in str(error.value)
    assert _published_refs(channel.path) == []


def test_checkpoint_selection_uses_trainer_rules_and_pins_symlink(tmp_path):
    import torch
    from specforge.training.checkpoint import CheckpointManager
    from specforge.training.online_resume import read_producer_checkpoint

    for step in (2, 4):
        directory = tmp_path / f'run-step{step}'
        directory.mkdir()
        torch.save({'global_step': step}, directory / 'training_state.pt')
    latest = tmp_path / 'run-latest'
    latest.symlink_to('run-step2')
    path, state = read_producer_checkpoint(str(tmp_path))
    assert path == CheckpointManager.resolve_resume_dir(str(tmp_path))
    assert state['global_step'] == 4
    # Explicit old-step links intentionally select the old step; freeze the
    # physical directory so moving the link cannot change the consumer target.
    path, state = read_producer_checkpoint('file://' + str(latest))
    latest.unlink()
    latest.symlink_to('run-step4')
    assert path == str(tmp_path / 'run-step2')
    assert CheckpointManager.read_resume_state(path, require_full_state=False)['global_step'] == state['global_step'] == 2


@pytest.mark.parametrize('total', [8, 9])
def test_completed_cursor_includes_settled_tail_but_rejects_invalid_state(total):
    plan = prompt_plan([{'task_id': str(i)} for i in range(total)], seed=42,
                       epochs=1, dp_size=2, batch_size=1, accumulation_steps=2)
    state = dict(global_step=2, epoch=1, epoch_batch=0, epoch_samples=0,
                 world_size=2, batch_size=1, accumulation_steps=2,
                 **{PLAN_KEY: plan})
    assert validate_resume_plan(state, plan, []) == total
    for change in ({'global_step': 1}, {'global_step': 3}, {'epoch_batch': 1},
                   {'epoch_samples': 1}, {'epoch': 2}):
        with pytest.raises(ValueError):
            resume_cursor(dict(state, **change), plan)
