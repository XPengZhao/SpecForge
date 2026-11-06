"""Download local datasets for sweep_dspark_checkpoints.py --data-dir."""
import argparse
import gzip
import json
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlencode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--insecure", action="store_true", help="Skip TLS verification for dataset downloads")
    parser.add_argument("--regen", type=Path, help="Optionally link an existing regen JSONL")
    args = parser.parse_args()
    dest = args.data_dir.resolve()
    (dest / "gsm8k").mkdir(parents=True, exist_ok=True)

    def fetch(url):
        cmd = ["curl", "-fsSL", "--connect-timeout", "15", "--max-time", "180", "--retry", "2"]
        if args.insecure:
            cmd.append("--insecure")
        return subprocess.check_output([*cmd, url])

    def save(name, rows, count, fields):
        if len(rows) != count or not all(all(field in row for field in fields) for row in rows):
            raise ValueError(f"Unexpected contents for {name} (expected {count} rows with {fields})")
        path = dest / (name + ".jsonl")
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        Path(f.name).replace(path)
        print(f"{path}: {count}", flush=True)

    base = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/ci-datasets/gsm8k"
    for split, count in (("train", 7473), ("test", 1319)):
        rows = fetch(f"{base}/{split}.jsonl").splitlines()
        save("gsm8k/" + split, [json.loads(r) for r in rows if r.strip()], count, ("question", "answer"))

    url = "https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz"
    rows = gzip.decompress(fetch(url)).splitlines()
    save("humaneval", [json.loads(r) for r in rows if r.strip()], 164, ("task_id", "prompt", "entry_point"))
    url = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
    save("mtbench", [json.loads(r) for r in fetch(url).splitlines() if r.strip()], 80, ("question_id", "turns"))

    for name, repo, split, count, fields in (
        ("math500", "HuggingFaceH4/MATH-500", "test", 500, ("problem", "answer")),
        ("aime", "Maxwell-Jia/AIME_2024", "train", 30, ("Problem", "Answer")),
    ):
        rows = []
        for offset in range(0, count, 100):
            length = min(100, count - offset)
            query = urlencode(dict(dataset=repo, config="default", split=split, offset=offset, length=length))
            page = json.loads(fetch("https://datasets-server.huggingface.co/rows?" + query))
            if [item["row_idx"] for item in page["rows"]] != list(range(offset, offset + length)):
                raise ValueError(f"Incomplete or out-of-order {name} page at {offset}")
            rows.extend(item["row"] for item in page["rows"])
        save(name, rows, count, fields)

    if args.regen is None:
        return
    source = args.regen.expanduser().resolve(strict=True)
    link = dest / "regen.jsonl"
    if not link.exists():
        link.symlink_to(source)
    elif not link.samefile(source):
        raise ValueError(f"{link} already points to another dataset")
    print(f"{link} -> {source}")


if __name__ == "__main__":
    main()
