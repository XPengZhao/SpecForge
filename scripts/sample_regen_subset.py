"""Freeze a uniform 10% sample of a regenerated training JSONL, preserving its IDs."""

import argparse
import hashlib
import json
import random
import tempfile
from pathlib import Path


def sample(source: Path, output: Path, seed: int) -> None:
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    count = 0
    digest = hashlib.sha256()
    with source.open("rb") as rows:
        for raw in rows:
            digest.update(raw)
            if raw.strip():
                count += 1
    size = count // 10
    if not size:
        raise ValueError("Need at least ten records to take a 10% sample")
    chosen = set(random.Random(seed).sample(range(count), size))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        bundle = Path(staging) / "subset"
        bundle.mkdir()
        check = hashlib.sha256()
        subset_hash = hashlib.sha256()
        written = 0
        index = 0
        with (
            source.open("rb") as rows,
            (bundle / "train.jsonl").open("wb") as train,
            (bundle / "ids.jsonl").open("w", encoding="utf-8") as ids,
        ):
            for line_number, raw in enumerate(rows, start=1):
                check.update(raw)
                if not raw.strip():
                    continue
                if index in chosen:
                    row = json.loads(raw)
                    if not isinstance(row, dict):
                        raise ValueError(f"Expected a JSON object at source line {line_number}")
                    written += 1
                    train.write(raw)
                    subset_hash.update(raw)
                    print(
                        json.dumps(
                            {
                                "id": row.get("id"),
                                "source_line": line_number,
                                "sample_id": f"line:{line_number}",
                            },
                            ensure_ascii=False,
                        ),
                        file=ids,
                    )
                index += 1
        if check.digest() != digest.digest() or written != size:
            raise ValueError("Source changed during sampling")
        manifest = {
            "source": str(source.resolve()),
            "source_sha256": digest.hexdigest(),
            "sample_id_scope": "1-based physical line in source_sha256",
            "source_rows": count,
            "sample_rows": size,
            "seed": seed,
            "algorithm": "uniform random.sample without replacement, floor(N/10), source order",
            "train_sha256": subset_hash.hexdigest(),
        }
        (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        bundle.rename(output)
    print(f"Saved {size:,} of {count:,} records to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sample(args.source, args.output_dir, args.seed)
