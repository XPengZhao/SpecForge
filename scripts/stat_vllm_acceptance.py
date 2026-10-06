"""Measure speculative acceptance using differences between /metrics snapshots."""
import argparse
import json
from pathlib import Path
import re
import urllib.request

FIELDS = {
    "vllm:spec_decode_num_drafts_total": "rounds",
    "vllm:spec_decode_num_draft_tokens_total": "drafted",
    "vllm:spec_decode_num_accepted_tokens_total": "accepted",
    "vllm:spec_decode_num_accepted_tokens_per_pos_total": "position",
}


def parse_metrics(body):
    series = {}
    for line in body.splitlines():
        match = re.fullmatch(r'([^\s{]+)(\{.*\})?\s+([0-9.eE+-]+)(?:\s+\d+)?', line)
        if not match or match[1] not in FIELDS:
            continue
        name, labels, raw = match.groups()
        label_pairs = re.findall(r'(\w+)="((?:\\.|[^"\\])*)"', labels or "")
        key = name + json.dumps(sorted(label_pairs), separators=(",", ":"))
        position = dict(label_pairs).get("position")
        value = float(raw)
        if value < 0 or not value.is_integer():
            raise ValueError("Invalid counter value")
        if FIELDS[name] == "position" and position is None:
            raise ValueError("Missing position label")
        series[key] = dict(field=FIELDS[name], position=int(position) if position is not None else None,
                           value=int(value))
    if not series:
        raise ValueError("No speculative counters found: is this a DSpark service?")
    return series


def subtract(before, after):
    if before["url"] != after["url"]:
        raise ValueError("Snapshots came from different URLs")
    if set(before["series"]) - set(after["series"]):
        raise ValueError("Counter series disappeared; check service restart/configuration")
    totals = dict(rounds=0, drafted=0, accepted=0)
    positions = {}
    for key, row in after["series"].items():
        delta = row["value"] - before["series"].get(key, {}).get("value", 0)
        if delta < 0:
            raise ValueError("Counters decreased: the service may have restarted")
        if row["field"] == "position":
            p = row["position"]
            positions[p] = positions.get(p, 0) + delta
        else:
            totals[row["field"]] += delta
    rounds, drafted, accepted = (totals[key] for key in ("rounds", "drafted", "accepted"))
    if rounds <= 0 or drafted <= 0:
        raise ValueError("No new draft rounds/tokens between snapshots")
    if accepted > drafted:
        raise ValueError("Accepted count exceeds drafted count")
    if not positions or sum(positions.values()) != accepted:
        raise ValueError("Position counts do not sum to accepted tokens; wait for metrics to update")
    previous, rows = rounds, []
    for position, count in sorted(positions.items()):
        if count > previous:
            raise ValueError("Continuous acceptance counts are not nonincreasing")
        rows.append(dict(position=position + 1, accepted=count, continuous_rate=count / rounds,
                         conditional_rate=count / previous if previous else None))
        previous = count
    return dict(**totals, mal=1 + accepted / rounds, acceptance_rate=accepted / drafted,
                positions=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("--url", default="http://127.0.0.1:8000")
    snapshot.add_argument("--output", required=True)
    diff = sub.add_parser("diff")
    diff.add_argument("--before", required=True)
    diff.add_argument("--after", required=True)
    diff.add_argument("--output")
    args = parser.parse_args()
    if args.command == "snapshot":
        path = Path(args.output)
        if path.exists():
            raise FileExistsError(f"Use a fresh snapshot filename: {path}")
        with urllib.request.urlopen(args.url.rstrip("/") + "/metrics", timeout=30) as response:
            series = parse_metrics(response.read().decode())
        with path.open("x") as stream:
            json.dump(dict(url=args.url.rstrip("/"), series=series), stream, indent=2)
        print(f"Saved {path}")
        return
    result = subtract(json.loads(Path(args.before).read_text()), json.loads(Path(args.after).read_text()))
    print(f"Draft rounds: {result['rounds']}\nDrafted tokens: {result['drafted']}\nAccepted tokens: {result['accepted']}")
    print(f"MAL: {result['mal']:.4f}\nAcceptance rate: {result['acceptance_rate']:.2%}")
    print("Position  Accepted  Continuous rate  Conditional rate")
    for row in result["positions"]:
        conditional = f"{row['conditional_rate']:.2%}" if row["conditional_rate"] is not None else "N/A"
        print(f"{row['position']:8d}  {row['accepted']:8d}  {row['continuous_rate']:15.2%}  {conditional:>16}")
    if args.output:
        with Path(args.output).open("x") as stream:
            json.dump(result, stream, indent=2)


if __name__ == "__main__":
    main()
