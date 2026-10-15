"""Validate GPU ownership and the DSpark capture contract without CUDA."""

import json
from pathlib import Path
from unittest import mock

from specforge.application import resolve_run
from specforge.config import load_config
from specforge.launch_plan import build_launch_plan
from specforge.training.capture_contract import ServerCaptureContract


def test_managed_qwen3_4b_plan_keeps_capture_and_training_on_distinct_gpus(tmp_path):
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(
        str(root / "examples/configs/qwen3-4b-dspark-1server-dp7-disaggregated.yaml"),
        [f"deployment.disaggregated.control_dir={tmp_path / 'control'}"],
    )
    contract = ServerCaptureContract(
        method="dflash",
        aux_layer_ids=(1, 9, 17, 25, 33),
        target_hidden_size=2560,
        target_vocab_size=151936,
        draft_vocab_size=151936,
    )
    with mock.patch(
        "specforge.training.capture_contract.resolve_server_capture_contract",
        return_value=contract,
    ):
        plan = build_launch_plan(
            cfg,
            config_path=str(
                root / "examples/configs/qwen3-4b-dspark-1server-dp7-disaggregated.yaml"
            ),
            algorithm=resolve_run(cfg).algorithm,
        )
    assert plan.kind == "managed_supervisor"
    producer, consumer = plan.commands
    assert consumer.env["CUDA_VISIBLE_DEVICES"] == "1,2,3,4,5,6,7"
    assert consumer.argv[consumer.argv.index("--nproc_per_node") + 1] == "7"
    capture = plan.services[1].command
    assert capture.env["MOONCAKE_GLOBAL_SEGMENT_SIZE"] == str(32 << 30)
    assert capture.env["CUDA_VISIBLE_DEVICES"] == "0"
    assert capture.argv[capture.argv.index("--spec-capture-method") + 1] == "dflash"
    assert capture.argv[capture.argv.index("--context-length") + 1] == "4103"
    assert "--disable-radix-cache" in capture.argv
    assert capture.argv[capture.argv.index("--chunked-prefill-size") + 1] == "-1"


def test_dspark_recipe_uses_full_vocab_and_original_offline_objective():
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(
        str(root / "examples/configs/qwen3-4b-dspark-1server-dp7-disaggregated.yaml")
    )
    offline = load_config(
        str(root / "examples/configs/qwen3-4b-dspark-deepspec-offline.yaml")
    )
    resolved = resolve_run(cfg)
    assert resolved.algorithm.name == "dspark"
    assert not cfg.model.vocab_mapping_path
    assert not resolved.algorithm.providers.vocab_mapping_modes
    stream = resolved.algorithm.providers.server_streaming_for("text")
    assert stream.capture_method == "dflash"
    assert stream.layout.last_hidden_feature == "target_last_hidden_states"
    draft = json.loads((root / cfg.model.draft_model_config).read_text())
    assert draft["architectures"] == ["DSparkDraftModel"]
    assert draft["dflash_config"]["target_layer_ids"] == [1, 9, 17, 25, 33]
    for field in (
        "dspark_loss_mode",
        "dspark_ce_loss_alpha",
        "dspark_l1_loss_alpha",
        "dspark_confidence_head_alpha",
        "loss_decay_gamma",
        "num_anchors",
    ):
        assert getattr(cfg.training, field) == getattr(offline.training, field)
