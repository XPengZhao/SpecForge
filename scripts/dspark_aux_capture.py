"""Isolated live backend workers for diagnose_dspark_prefixes.py (local Linux GPU)."""

import importlib.metadata
import inspect
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path

import torch


def read(path):
    return json.loads(path.read_text())


def ports(n):
    with ExitStack() as stack:
        sockets = [stack.enter_context(socket.socket()) for _ in range(n)]
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        return [sock.getsockname()[1] for sock in sockets]


@contextmanager
def service(command, log, ready):
    with log.open("a") as stream:
        proc = subprocess.Popen(
            command, stdout=stream, stderr=stream, start_new_session=True
        )
        try:
            deadline = time.monotonic() + 600
            while proc.poll() is None and time.monotonic() < deadline:
                if ready():
                    break
                time.sleep(1)
            else:
                raise RuntimeError(f"Service failed to start. See {log}")
            yield
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


def write_capture(dest, ids, aux, final, layers, width):
    if aux.shape != (len(ids), len(layers) * width):
        raise ValueError(f"{dest}: unexpected aux shape {aux.shape}")
    if final is not None and final.shape != (len(ids), width):
        raise ValueError(f"{dest}: unexpected final shape {final.shape}")
    data = {"ids": ids, "layers": layers, "aux": aux.cpu().contiguous().clone()}
    if final is not None:
        data["final"] = final.cpu().contiguous().clone()
    if any(
        not t.isfinite().all() for t in data.values() if isinstance(t, torch.Tensor)
    ):
        raise ValueError(f"{dest}: nonfinite capture")
    tmp = dest.with_suffix(".tmp")
    torch.save(data, tmp)
    tmp.replace(dest)


def trainer(out, manifest, pending, layers, width):
    import requests
    from specforge.algorithms.common.dflash_family_data import build_dspark_collator
    from specforge.algorithms.dspark.providers import algorithm_providers
    from specforge.inference.adapters.server_capture import (
        SGLangServerCaptureAdapter,
        ServerCaptureSchema,
    )
    from specforge.inference.capture import CaptureConfig
    from specforge.runtime.contracts import PromptTask, SampleRef
    from specforge.runtime.data_plane.feature_dataloader import FeatureDataLoader
    from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore

    rpc, metadata, metrics, port = ports(4)
    url = f"http://127.0.0.1:{port}"
    master = f"127.0.0.1:{rpc}"
    meta_url = f"http://127.0.0.1:{metadata}/metadata"
    os.environ.update(
        MOONCAKE_LOCAL_HOSTNAME="127.0.0.1",
        MOONCAKE_PROTOCOL="tcp",
        MOONCAKE_MASTER_SERVER_ADDR=master,
        MOONCAKE_METADATA_SERVER=meta_url,
    )

    def master_ready():
        try:
            with socket.create_connection(("127.0.0.1", rpc), timeout=1):
                return True
        except OSError:
            return False

    def teacher_ready():
        try:
            return requests.get(url + "/health", timeout=2).ok
        except requests.RequestException:
            return False

    with ExitStack() as stack:
        stack.enter_context(
            service(
                [
                    "mooncake_master",
                    "--enable_http_metadata_server=true",
                    "--http_metadata_server_host=127.0.0.1",
                    f"--rpc_port={rpc}",
                    f"--http_metadata_server_port={metadata}",
                    f"--metrics_port={metrics}",
                ],
                out / "mooncake.log",
                master_ready,
            )
        )
        stack.enter_context(
            service(
                [
                    sys.executable,
                    "-m",
                    "sglang.launch_server",
                    "--model-path",
                    manifest["target_dir"],
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--skip-tokenizer-init",
                    "--attention-backend",
                    manifest["aux_comparison"]["sglang_attention_backend"],
                    "--enable-spec-capture",
                    "--spec-capture-method",
                    "dflash",
                    "--spec-capture-aux-layer-ids",
                    *map(str, layers),
                    "--chunked-prefill-size",
                    "-1",
                    "--disable-radix-cache",
                    "--dtype",
                    "bfloat16",
                    "--context-length",
                    str(manifest["max_length"] + 64),
                ],
                out / "sglang.log",
                teacher_ready,
            )
        )
        run = uuid.uuid4().hex
        store = MooncakeFeatureStore(
            store_id=run,
            setup_kwargs=dict(
                local_hostname="127.0.0.1",
                metadata_server=meta_url,
                master_server_addr=master,
            ),
        )
        layout = algorithm_providers().server_streaming_for("text").layout
        schema = ServerCaptureSchema(
            layout.aux_feature,
            layout.last_hidden_feature,
            layout.passthrough,
            layout.attention_mask_feature,
        )
        adapter = SGLangServerCaptureAdapter(
            url, store, run_id=run, algorithm="dspark", schema=schema
        )
        contract = CaptureConfig.from_strategy(
            {"input_ids", "loss_mask", "hidden_states", "target_last_hidden_states"},
            layers,
            target_repr="hidden_state",
            target_hidden_size=width,
        )
        for start in range(0, len(pending), 2):
            group = pending[start : start + 2]
            tasks = [
                PromptTask(
                    str(s["row"]),
                    run,
                    "diagnosis",
                    {
                        "input_ids": s["ids"],
                        "loss_mask": (
                            s["loss_mask"]
                            if "loss_mask" in s
                            else [
                                int(i >= s["prompt_length"])
                                for i in range(len(s["ids"]))
                            ]
                        ),
                    },
                    len(s["ids"]),
                )
                for s, _ in group
            ]
            refs = adapter.produce_refs(tasks, capture=contract)
            if not all(isinstance(ref, SampleRef) for ref in refs):
                raise RuntimeError(f"Teacher capture failed: {refs}")
            loader = FeatureDataLoader(
                store,
                refs=refs,
                batch_size=len(refs),
                strategy="dspark",
                collate_fn=build_dspark_collator(),
            )
            batch = next(iter(loader))
            for i, (sample, dest) in enumerate(group):
                n = len(sample["ids"])
                ids = batch.tensors["input_ids"][i, :n].tolist()
                if (
                    ids != sample["ids"]
                    or batch.sample_ids[i] != f"{run}:{sample['row']}"
                ):
                    raise ValueError(f"{dest}: trainer tokens or sample ID changed")
                write_capture(
                    dest,
                    ids,
                    batch.tensors["hidden_states"][i, :n],
                    batch.tensors["target_last_hidden_states"][i, :n],
                    layers,
                    width,
                )


def vllm_capture(out, manifest, pending, layers, width):
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_USE_V2_MODEL_RUNNER="1")
    from vllm import LLM, SamplingParams
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
    from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
        get_eagle3_aux_layers_from_config,
    )

    draft_path = manifest["aux_comparison"]["export"]
    block_size = next(iter(manifest["drafts"].values()))["config_values"]["block_size"]
    engine = LLM(
        model=manifest["target_dir"],
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=manifest["aux_comparison"]["vllm_eager"],
        max_num_seqs=1,
        enable_prefix_caching=False,
        max_num_batched_tokens=manifest["max_length"] + 64,
        max_model_len=manifest["max_length"] + 64,
        speculative_config={
            "method": "dspark",
            "attention_backend": "FLASH_ATTN",
            "model": draft_path,
            "enable_adaptive_verification": False,
            "num_speculative_tokens": block_size,
        },
    )
    original = DSparkSpeculator.propose
    signature = inspect.signature(original)
    observed = None

    def capture(self, *args, **kwargs):
        nonlocal observed
        if observed is None:
            call = signature.bind(self, *args, **kwargs).arguments
            if call.get("dummy_run", False):
                return original(self, *args, **kwargs)
            batch = call["input_batch"]
            n = batch.num_tokens
            ids = batch.input_ids[:n].tolist()
            positions = batch.positions[:n].tolist()
            resolved = get_eagle3_aux_layers_from_config(self.speculative_config)
            if resolved != tuple(i + 1 for i in layers):
                raise ValueError(
                    f"vLLM resolved capture boundaries {resolved}, expected layers {layers}"
                )
            aux = call["aux_hidden_states"]
            if (
                aux is None
                or len(aux) != len(layers)
                or any(h[:n].shape != (n, width) for h in aux)
            ):
                raise ValueError("Unexpected vLLM DSpark aux layer shapes")
            if ids != sample["ids"] or positions != list(range(len(ids))):
                raise ValueError(
                    f"row {sample['row']}: vLLM first proposal token/position mismatch"
                )
            observed = torch.cat([h[:n] for h in aux], dim=-1).detach().cpu()
        return original(self, *args, **kwargs)

    DSparkSpeculator.propose = capture
    try:
        for sample, dest in pending:
            observed = None
            engine.generate(
                [{"prompt_token_ids": sample["ids"]}],
                SamplingParams(temperature=0, max_tokens=2, ignore_eos=True),
                use_tqdm=False,
            )
            if observed is None:
                raise RuntimeError("vLLM did not call the in-process DSpark proposer")
            write_capture(dest, sample["ids"], observed, None, layers, width)
    finally:
        DSparkSpeculator.propose = original


def main():
    source, output = sys.argv[1:]
    out = Path(output)
    manifest = read(out / "manifest.json")
    layers = manifest["aux_comparison"]["layers"]
    width = read(Path(manifest["target_dir"]) / "config.json")["hidden_size"]
    directory = out / "aux-comparison" / source
    directory.mkdir(parents=True, exist_ok=True)
    samples = read(out / "prefixes.json")
    if source == "trainer" and "perfectblend" in manifest:
        samples += read(out / "perfectblend-prefixes.json")
    pending = [(s, directory / f"row-{s['row']}.pt") for s in samples]
    pending = [(s, p) for s, p in pending if not p.exists()]
    if pending:
        version = {
            "package": importlib.metadata.version(
                "sglang" if source == "trainer" else "vllm"
            ),
            "torch": str(torch.__version__),
            "python": sys.executable,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        }
        version_file = directory / "runtime.json"
        if version_file.exists() and read(version_file) != version:
            raise ValueError(f"{source} runtime changed during resumed capture")
        version_file.write_text(json.dumps(version, indent=2))
        print(f"Capturing {source}: {len(pending)} questions", flush=True)
        ({"trainer": trainer, "vllm": vllm_capture}[source])(
            out, manifest, pending, layers, width
        )


if __name__ == "__main__":
    main()
