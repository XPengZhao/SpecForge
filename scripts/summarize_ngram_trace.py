"""Summarize real verification traces. Run on the server or copied trace files.

Top-K residuals are discovery data only. Direction consistency is reported only
for fixed candidate sets supplied to the server before the run.
"""

import argparse
import csv
import hashlib
import json
import re
import sqlite3
from pathlib import Path

if __package__:
    from .ngram_contributions import ContributionSummary
else:
    from ngram_contributions import ContributionSummary


def records(files, tasks, prefix):
    pattern = re.compile(re.escape(prefix) + r"\d{9}(?!\d)")
    seen = {}
    for path in files:
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("type") != "event":
                    continue
                match = pattern.search(row["request_id"])
                if not match or match[0] not in tasks:
                    continue
                task = tasks[match[0]]
                key = (match[0], row["anchor_position"], row["position"])
                # Multiple TP workers can report the same verification. Fail on
                # disagreement instead of quietly counting a reordered row twice.
                signature = hashlib.sha256(
                    json.dumps(
                        {k: v for k, v in row.items() if k != "request_id"},
                        sort_keys=True,
                    ).encode()
                ).digest()
                if key in seen:
                    if seen[key] != signature:
                        raise ValueError(f"Conflicting trace rows for {key}")
                    continue
                seen[key] = signature
                row["prompt_hash"] = task.get(
                    "prompt_family_sha256", task["prompt_sha256"]
                )
                if "turn" in task:
                    row["turn"] = task["turn"]
                if "dataset" in task:
                    row["dataset"] = task["dataset"]
                row["trace_request_id"] = match[0]
                yield row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--trace-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--min-count", type=int, default=10)
    p.add_argument("--min-prompts", type=int, default=3)
    p.add_argument(
        "--direction-epsilon",
        type=float,
        default=0.001,
        help="Meaningful delta A threshold; .001 = 0.1 percentage point",
    )
    p.add_argument(
        "--direction-min-prompt-events",
        type=int,
        default=3,
        help="Deprecated: ignored by event-based direction classification",
    )
    p.add_argument(
        "--direction-min-prompts",
        type=int,
        default=5,
        help="Deprecated: ignored by event-based direction classification",
    )
    p.add_argument("--max-candidates", type=int, default=20000)
    p.add_argument("--candidate-tokens", type=int, default=32)
    args = p.parse_args(argv)
    if (
        min(
            args.min_count, args.min_prompts, args.max_candidates, args.candidate_tokens
        )
        < 1
    ):
        p.error("thresholds must be positive")
    manifest = args.eval_dir / "trace_manifest.json"
    if manifest.exists():
        config = json.loads(manifest.read_text())
        evaluation = config
    else:
        config = json.loads((args.eval_dir / "config.json").read_text())
        evaluation = json.loads((args.eval_dir / "regen.json").read_text())
    if not evaluation["valid_measurement"]:
        raise ValueError("Client evaluation failed; do not aggregate as a valid run")
    tasks = {x["trace_request_id"]: x for x in config["tasks"]}
    files = sorted(args.trace_dir.glob("trace-*.jsonl"))
    if not files:
        raise ValueError("No server trace files: check V2 runner and trace environment")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    db = sqlite3.connect(args.output_dir / "counts.sqlite")
    db.executescript(
        """CREATE TABLE counts (gram TEXT,pos INTEGER,n INTEGER,accepted INTEGER,rejected INTEGER,l1 REAL,PRIMARY KEY(gram,pos));
        CREATE TABLE coverage (gram TEXT,pos INTEGER,prompt TEXT,PRIMARY KEY(gram,pos,prompt));"""
    )
    contributions = ContributionSummary(
        args.output_dir,
        direction_epsilon=args.direction_epsilon,
        direction_min_prompt_events=args.direction_min_prompt_events,
        direction_min_prompts=args.direction_min_prompts,
    )
    traced = set()
    total = reached = 0
    for row in records(files, tasks, config["trace_prefix"]):
        traced.add(row["trace_request_id"])
        total += 1
        contributions.add(row)
        if not row["reached"]:
            continue
        reached += 1
        for field in ("ngram2", "ngram3"):
            if not row[field]:
                continue
            gram = ",".join(map(str, row[field]))
            pos = row["position"]
            db.execute(
                "INSERT INTO counts VALUES (?,?,?,?,?,?) ON CONFLICT(gram,pos) DO UPDATE SET n=n+1,accepted=accepted+excluded.accepted,rejected=rejected+excluded.rejected,l1=l1+excluded.l1",
                (
                    gram,
                    pos,
                    1,
                    int(row["accepted"]),
                    int(row["first_rejection"]),
                    row["raw_l1"],
                ),
            )
            db.execute(
                "INSERT OR IGNORE INTO coverage VALUES (?,?,?)",
                (gram, pos, row["prompt_hash"]),
            )
        if total % 1000 == 0:
            db.commit()
    db.commit()
    if not total:
        raise ValueError(
            "No matching trace events; check request prefix and trace directory"
        )
    contribution_summary = contributions.finish(args.min_count, args.min_prompts)
    selected = {}
    query = """SELECT c.gram,c.pos,c.n,c.accepted,c.rejected,c.l1,count(*) FROM counts c JOIN coverage v ON c.gram=v.gram AND c.pos=v.pos
             GROUP BY c.gram,c.pos HAVING c.n>=? AND count(*)>=? ORDER BY c.rejected DESC,c.n DESC LIMIT ?"""
    for g, pos, n, a, r, l1, coverage in db.execute(
        query, (args.min_count, args.min_prompts, args.max_candidates)
    ):
        selected[(g, pos)] = dict(
            ngram_token_ids=list(map(int, g.split(","))),
            position=pos,
            reached=n,
            accepted=a,
            first_rejections=r,
            prompts=coverage,
            conditional_acceptance=a / n,
            mean_raw_l1=l1 / n,
        )
    db.close()
    discovery = {}
    fixed = {}
    for row in records(files, tasks, config["trace_prefix"]):
        if not row["reached"]:
            continue
        for field in ("ngram2", "ngram3"):
            if not row[field]:
                continue
            key = (",".join(map(str, row[field])), row["position"])
            if key not in selected:
                continue
            # Use group A only for candidate discovery, preserving a held-out B.
            group = int(row["prompt_hash"][:16], 16) % 2
            if group == 0:
                scores = discovery.setdefault(key, {})
                for token, value in zip(row["top_token_ids"], row["top_residual"]):
                    scores[token] = max(scores.get(token, 0.0), abs(value))
                discovery[key] = dict(
                    sorted(scores.items(), key=lambda x: (-x[1], x[0]))[
                        : args.candidate_tokens
                    ]
                )
            for item in row["fixed"]:
                if item["ngram_token_ids"] != row[field]:
                    continue
                import numpy as np

                stat = fixed.setdefault(
                    (key, group),
                    dict(
                        ids=item["token_ids"],
                        n=0,
                        l1=0.0,
                        sum=np.zeros(len(item["token_ids"])),
                    ),
                )
                if stat["ids"] != item["token_ids"]:
                    raise ValueError("Mixed candidate sets in one run")
                stat["n"] += 1
                stat["l1"] += row["raw_l1"]
                stat["sum"] += item["residual"]
    with (args.output_dir / "candidates.jsonl").open("x") as out:
        for key, scores in discovery.items():
            out.write(
                json.dumps(
                    dict(
                        ngram_token_ids=selected[key]["ngram_token_ids"],
                        position=key[1],
                        token_ids=sorted(scores),
                    )
                )
                + "\n"
            )
    with (args.output_dir / "results.jsonl").open("x") as out:
        for key, row in selected.items():
            for group, name in ((0, "discovery"), (1, "heldout")):
                stat = fixed.get((key, group))
                if stat:
                    row[name] = dict(
                        count=stat["n"],
                        token_ids=stat["ids"],
                        mean_residual=(stat["sum"] / stat["n"]).tolist(),
                        selected_consistency=float(abs(stat["sum"]).sum())
                        / max(stat["l1"], 1e-30),
                    )
            out.write(json.dumps(row) + "\n")
    with (args.output_dir / "results.csv").open("x", newline="") as out:
        cols = [
            "ngram_token_ids",
            "position",
            "reached",
            "accepted",
            "first_rejections",
            "prompts",
            "conditional_acceptance",
            "mean_raw_l1",
        ]
        writer = csv.DictWriter(out, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected.values())
    summary = dict(
        contributions=contribution_summary,
        requests=len(tasks),
        requests_with_drafts=len(traced),
        events=total,
        reached_events=reached,
        rows=len(selected),
        fixed_candidate_statistics=bool(fixed),
        note="Legacy residual statistics use reached positions; paired contributions separate reached and after-rejection cohorts. Sparse top-K is not a consistency estimate.",
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
