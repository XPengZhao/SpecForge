"""DeepSpec v2 binary protocol and SpecForge read/write integration."""

import json
import struct
import tempfile
import unittest
from pathlib import Path

import torch

from specforge.algorithms.common.dflash_family_data import (
    build_dspark_collator, build_offline_dspark_reader, normalize_offline_dspark_sample,
)
from specforge.runtime.data_plane.deepspec_cache import DeepSpecCacheReader, TOKEN_ALIGNED_MASK
from specforge.runtime.data_plane.disagg_ingest import ingest_offline_features
from specforge.runtime.data_plane.feature_dataloader import FeatureDataLoader
from specforge.runtime.data_plane.feature_store import LocalFeatureStore
from specforge.runtime.data_plane.ref_serialization import ref_from_dict, ref_to_dict


def write_fixture(root):
    """Independent encoder following DeepSpec's <QIIQQQQQ record protocol."""
    expected = []
    records = []
    for i, length in enumerate((5, 3)):
        ids = torch.arange(length, dtype=torch.int32) + i * 10
        mask = torch.ones(length, dtype=torch.uint8)
        mask[0] = 0
        aux = torch.arange(length * 8, dtype=torch.float32).reshape(length, 8).bfloat16()
        last = -aux[:, :4].contiguous()
        fields = (ids, torch.ones_like(mask), mask, aux, last)
        data, offsets = bytearray(), []
        for field in fields:
            offsets.append(len(data))
            data.extend(field.contiguous().view(torch.uint8).numpy().tobytes())
        (root / f'shard-{i:05d}.bin').write_bytes(data)
        records.append(struct.pack('<QIIQQQQQ', i, i, length, *offsets))
        expected.append(dict(input_ids=ids.long(), loss_mask=mask,
                             aux_hidden_state=aux, hidden_state=last))
    (root / 'samples.idx').write_bytes(b''.join(records))
    manifest = dict(version=2, num_samples=2, num_shards=2, hidden_size=4,
                    target_layer_ids=[1, 3], target_model_name_or_path='Qwen/Qwen3-4B',
                    hidden_dtype='bfloat16', token_dtype='int32', mask_dtype='uint8',
                    index_record_size=56,
                    shards=[dict(shard_id=i, file_name=f'shard-{i:05d}.bin') for i in range(2)])
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return expected


class TestDeepSpecCache(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.expected = write_fixture(self.root)

    def reader(self, **kw):
        return build_offline_dspark_reader(str(self.root), run_id='test', ttt_length=7,
                                          max_len=kw.get('max_len', 4096))

    def test_read_only_roundtrip_truncation_and_names(self):
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        store = LocalFeatureStore()
        for i, ref in enumerate(self.reader(max_len=4)):
            ref = ref_from_dict(json.loads(json.dumps(ref_to_dict(ref))))
            raw, handle = store.get(ref)
            store.release(handle)
            for key, tensor in self.expected[i].items():
                torch.testing.assert_close(raw[key], tensor[:4], rtol=0, atol=0)
            normalized = normalize_offline_dspark_sample(raw, 4)
            self.assertEqual(normalized['loss_mask'][0, -1].item(), 1)
            self.assertEqual(normalized['input_ids'].dtype, torch.int64)
            subset, handle = store.get(ref, names=['hidden_state'])
            store.release(handle)
            self.assertEqual(set(subset), {'hidden_state'})
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})
        self.assertEqual(self.reader().read(limit=0), [])

    def test_loader_padding_and_prefetch(self):
        for workers in (0, 2):
            loader = FeatureDataLoader(
                LocalFeatureStore(), refs=self.reader().read(), batch_size=2,
                strategy='dspark', num_workers=workers, drop_last=False,
                per_sample_transform=lambda raw: normalize_offline_dspark_sample(raw, 4096),
                collate_fn=build_dspark_collator(),
            )
            batch = next(iter(loader)).tensors
            self.assertEqual(tuple(batch['hidden_states'].shape), (2, 5, 8))
            self.assertEqual(batch['loss_mask'].tolist(), [[0, 1, 1, 1, 1], [0, 1, 1, 0, 0]])

    def test_store_write_and_native_reread_preserve_mask(self):
        dest = self.root / 'native'
        store = LocalFeatureStore(dump_dir=str(dest))
        refs = ingest_offline_features(
            store, str(self.root), algorithm_name='dspark',
            build_reader=build_offline_dspark_reader, limit=1,
        )
        raw, handle = store.get(refs[0]); store.release(handle)
        self.assertTrue(raw[TOKEN_ALIGNED_MASK].item())
        reader = build_offline_dspark_reader(str(dest), run_id='again', ttt_length=7, max_len=4096)
        ref = reader.read()[0]
        raw, handle = store.get(ref); store.release(handle)
        normalized = normalize_offline_dspark_sample(raw, 4096)
        self.assertEqual(normalized['loss_mask'][0, -1].item(), 1)

    def test_legacy_mask_unchanged(self):
        normalized = normalize_offline_dspark_sample(self.expected[0], 4096)
        self.assertEqual(normalized['loss_mask'][0, -1].item(), 0)
        self.assertEqual(self.expected[0]['loss_mask'][-1].item(), 1)

    def test_model_validation(self):
        reader = self.reader()
        reader.validate_model(hidden_size=4, target_layer_ids=[1, 3], target_model_path='Qwen/Qwen3-4B')
        for args in [dict(hidden_size=5, target_layer_ids=[1, 3]),
                     dict(hidden_size=4, target_layer_ids=[3, 1])]:
            with self.assertRaises(ValueError):
                reader.validate_model(**args, target_model_path='Qwen/Qwen3-4B')
        with self.assertRaises(ValueError):
            reader.validate_model(hidden_size=4, target_layer_ids=[1, 3], target_model_path='other')

    def test_bad_manifest_index_and_shard(self):
        path = self.root / 'manifest.json'
        original = json.loads(path.read_text())
        for key, value in [('version', 1), ('index_record_size', 64), ('hidden_dtype', 'float16'),
                           ('num_samples', 3), ('num_shards', 3), ('target_layer_ids', [3, 1])]:
            path.write_text(json.dumps({**original, key: value}))
            with self.assertRaises(ValueError):
                self.reader()
        path.write_text(json.dumps(original))
        reader = self.reader()
        refs = reader.read()
        (self.root / 'shard-00000.bin').write_bytes(b'')
        with self.assertRaises(ValueError):
            self.reader().read()
        with self.assertRaises(ValueError):
            LocalFeatureStore().get(refs[0])


if __name__ == '__main__':
    unittest.main()
