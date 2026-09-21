"""Validate and index DeepSpec's raw ngram sidecar without loading payloads."""
import struct
from pathlib import Path

RECORD = struct.Struct('<QIIQ')
KEY = 'ngram_embedding'


class NgramSidecar:
    def __init__(self, root, manifest):
        meta = manifest.get('extra_features', {}).get(KEY)
        if not meta:
            raise ValueError('Ngram MASK requires a DeepSpec raw ngram_embedding sidecar')
        expected = dict(version=1, dtype='bfloat16', index_record_format=RECORD.format,
                        index_record_size=RECORD.size, hidden_size=manifest['hidden_size'],
                        num_samples=manifest['num_samples'], token_alignment='same_position_as_input_ids',
                        feature_stage='raw_ngram_lookup_concat_before_key_value_projection')
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f'Ngram sidecar {key}: expected {value!r}, got {meta.get(key)!r}')
        root = Path(root).resolve()
        base = (root / meta['path']).resolve()
        self.index = (base / meta['index_file']).resolve()
        if not base.is_relative_to(root) or not self.index.is_relative_to(base):
            raise ValueError('Ngram sidecar path outside cache')
        if self.index.stat().st_size != meta['num_samples'] * RECORD.size:
            raise ValueError('Ngram index size does not match main cache')
        self.width = meta['hidden_size']
        self.shards = []
        for shard in meta['shards']:
            path = (base / shard['file_name']).resolve()
            if not path.is_relative_to(base) or path.stat().st_size != shard['num_bytes']:
                raise ValueError('Invalid ngram shard path or size')
            self.shards.append((str(path), shard['num_bytes']))

    def next_sample(self, stream, sample_id, seq_len):
        raw = stream.read(RECORD.size)
        if len(raw) != RECORD.size:
            raise ValueError('Truncated ngram index')
        sid, shard, length, offset = RECORD.unpack(raw)
        if sid != sample_id or length != seq_len or shard >= len(self.shards):
            raise ValueError('Ngram/main sample ID or sequence length mismatch')
        path, size = self.shards[shard]
        if offset % 2 or offset + length * self.width * 2 > size:
            raise ValueError('Ngram payload outside shard')
        return {'path': path, 'offset': offset}
