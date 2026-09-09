"""Read DeepSpec v2 target-cache shards without converting or copying the dataset."""

from __future__ import annotations

import json
import struct
from itertools import islice
from pathlib import Path

import torch

from specforge.runtime.contracts import FeatureSpec, SampleRef

INDEX_RECORD = struct.Struct("<QIIQQQQQ")
TOKEN_ALIGNED_MASK = "loss_mask_is_token_aligned"
FEATURE_KEYS = ("input_ids", "loss_mask", "aux_hidden_state", "hidden_state", TOKEN_ALIGNED_MASK)


def is_deepspec_cache(path):
    return (Path(path).expanduser() / "manifest.json").is_file()


class DeepSpecCacheReader:
    """Scan only the small index; tensor data is fetched on demand by the store."""

    feature_keys = FEATURE_KEYS
    target_repr = "hidden_state"

    def __init__(self, path, *, run_id="offline", ttt_length=7, max_len=4096):
        self.root = Path(path).expanduser().resolve()
        self.run_id, self.ttt_length, self.max_len = run_id, ttt_length, int(max_len)
        if self.max_len <= 0:
            raise ValueError("max_len must be positive")
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        m = self.manifest
        for key, expected in {
            "version": 2, "index_record_size": INDEX_RECORD.size,
            "hidden_dtype": "bfloat16", "token_dtype": "int32", "mask_dtype": "uint8",
        }.items():
            if m.get(key) != expected:
                raise ValueError(f"DeepSpec {key}: expected {expected!r}, got {m.get(key)!r}")
        self.hidden_size = int(m["hidden_size"])
        self.layers = m["target_layer_ids"]
        if (self.hidden_size <= 0 or not self.layers
                or any(type(x) is not int or x < 0 for x in self.layers)
                or sorted(set(self.layers)) != self.layers):
            raise ValueError("Invalid DeepSpec hidden_size or target_layer_ids")
        if not isinstance(m.get("target_model_name_or_path"), str) or not m["target_model_name_or_path"]:
            raise ValueError("DeepSpec target_model_name_or_path is required")
        self.num_samples = int(m["num_samples"])
        if self.num_samples <= 0:
            raise ValueError("DeepSpec cache is empty")
        self.index_path = self.root / "samples.idx"
        if self.index_path.stat().st_size != self.num_samples * INDEX_RECORD.size:
            raise ValueError("DeepSpec samples.idx size does not match num_samples")
        shards = m["shards"]
        if len(shards) != int(m["num_shards"]):
            raise ValueError("DeepSpec num_shards does not match shard list")
        self.shards = []
        for i, shard in enumerate(shards):
            path = (self.root / shard["file_name"]).resolve()
            if shard["shard_id"] != i or not path.is_relative_to(self.root):
                raise ValueError("DeepSpec shard ids must be contiguous and paths inside cache")
            self.shards.append((str(path), path.stat().st_size))

    def validate_model(self, *, hidden_size, target_layer_ids, target_model_path):
        if int(hidden_size) != self.hidden_size or list(target_layer_ids) != self.layers:
            raise ValueError("DeepSpec cache hidden size / target layer order does not match draft")
        actual = self.manifest["target_model_name_or_path"]
        if str(target_model_path).rstrip("/") != actual.rstrip("/"):
            raise ValueError(
                f"DeepSpec cache target is {actual!r}, training target is {target_model_path!r}; "
                "use the same target checkpoint identifier as capture"
            )

    def __iter__(self):
        width = len(self.layers) * self.hidden_size
        keys = {name: name for name in FEATURE_KEYS}
        specs_by_length = {}
        with self.index_path.open("rb") as index:
            for i in range(self.num_samples):
                record = index.read(INDEX_RECORD.size)
                if len(record) != INDEX_RECORD.size:
                    raise ValueError("Truncated DeepSpec index")
                sample_id, shard_id, length, *offsets = INDEX_RECORD.unpack(record)
                if sample_id != i or length <= 0 or shard_id >= len(self.shards):
                    raise ValueError(f"Invalid DeepSpec index record {i}")
                path, shard_size = self.shards[shard_id]
                sizes = (4 * length, length, length, 2 * length * width,
                         2 * length * self.hidden_size)
                if any(offset + size > shard_size for offset, size in zip(offsets, sizes)):
                    raise ValueError(f"DeepSpec sample {i} extends beyond shard")
                if any(offsets[j] + sizes[j] > offsets[j + 1] for j in range(4)):
                    raise ValueError(f"Overlapping DeepSpec fields in sample {i}")
                n = min(length, self.max_len)
                specs = specs_by_length.get(n)
                if specs is None:
                    specs = {
                        "input_ids": FeatureSpec("input_ids", (n,), "int64"),
                        "loss_mask": FeatureSpec("loss_mask", (n,), "uint8"),
                        "aux_hidden_state": FeatureSpec("aux_hidden_state", (n, width), "bfloat16"),
                        "hidden_state": FeatureSpec("hidden_state", (n, self.hidden_size), "bfloat16"),
                        TOKEN_ALIGNED_MASK: FeatureSpec(TOKEN_ALIGNED_MASK, (), "bool"),
                    }
                    specs_by_length[n] = specs
                yield SampleRef(
                    sample_id=f"{self.run_id}:{i:08d}", run_id=self.run_id,
                    source_task_id=None, feature_store_uri=f"deepspec://{path}",
                    feature_keys=keys, feature_specs=specs, strategy="dspark",
                    target_model_version=self.manifest["target_model_name_or_path"],
                    num_tokens=n, estimated_bytes=n * (9 + 2 * (width + self.hidden_size)) + 1,
                    metadata={"format": "deepspec_v2", "target_repr": self.target_repr,
                              "ttt_length": self.ttt_length, "max_len": self.max_len,
                              "file_index": i, "offsets": offsets},
                )

    def read(self, limit=None):
        return list(islice(self, limit)) if limit is not None else list(self)


def read_deepspec_features(ref, names):
    """Read owned tensors with independent file handles, safe for loader threads."""
    path = ref.feature_store_uri.removeprefix("deepspec://")
    offsets = ref.metadata["offsets"]
    field_offsets = dict(zip(FEATURE_KEYS[:4], (offsets[0], offsets[2], offsets[3], offsets[4])))
    storage_dtypes = {"input_ids": torch.int32, "loss_mask": torch.uint8,
                      "aux_hidden_state": torch.bfloat16, "hidden_state": torch.bfloat16}
    result = {}
    with open(path, "rb") as stream:
        for name in names:
            if name == TOKEN_ALIGNED_MASK:
                result[name] = torch.tensor(True)
                continue
            spec = ref.feature_specs[name]
            dtype = storage_dtypes[name]
            count = 1
            for size in spec.shape:
                count *= size
            nbytes = count * torch.empty((), dtype=dtype).element_size()
            stream.seek(field_offsets[name])
            data = bytearray(stream.read(nbytes))
            if len(data) != nbytes:
                raise ValueError(f"Truncated DeepSpec shard: {path}, field {name}")
            tensor = torch.frombuffer(data, dtype=dtype).reshape(spec.shape)
            result[name] = tensor.long() if name == "input_ids" else tensor
    return result
