"""Event-based Engram effects per n-gram, with context details for inspection."""

import csv
import json
from collections import Counter, defaultdict
from itertools import groupby

LABELS = {
    "mostly_helpful": "多数有益",
    "mostly_harmful": "多数有害",
    "mixed_effect": "正负混合",
    "weak_effect": "作用弱",
    "insufficient_data": "样本不足",
    "inconclusive": "倾向不明确",
}


def classify(row, min_count, epsilon):
    if row["events"] < min_count:
        return "insufficient_data"
    if row["mean_abs_delta_a"] <= epsilon:
        return "weak_effect"
    if (
        row["improved_events"] >= 2
        and row["harmed_events"] >= 2
        and row["positive_event_fraction"] >= 0.2
        and row["negative_event_fraction"] >= 0.2
    ):
        return "mixed_effect"
    if row["positive_event_fraction"] >= 0.7 and row["mean_delta_a"] > epsilon:
        return "mostly_helpful"
    if row["negative_event_fraction"] >= 0.7 and row["mean_delta_a"] < -epsilon:
        return "mostly_harmful"
    return "inconclusive"


def write_context_details(db, directory):
    columns = [
        "ngram_token_ids",
        "position",
        "cohort",
        "turn",
        "prompt_hash",
        "events",
        "mean_delta_a",
        "g_plus",
        "g_minus",
    ]
    with (directory / "ngram_prompt_contributions.csv").open("x", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        for gram, pos, cohort, turn, prompt, n, delta, gain, loss, _, _ in db.execute(
            "SELECT * FROM direction_prompts ORDER BY gram,position,cohort,turn,prompt"
        ):
            writer.writerow(
                dict(
                    zip(
                        columns,
                        [
                            gram,
                            pos,
                            cohort,
                            turn,
                            prompt,
                            n,
                            delta / n,
                            gain / n,
                            loss / n,
                        ],
                    )
                )
            )
    columns = [
        "ngram_token_ids",
        "position",
        "cohort",
        "turn",
        "events",
        "prompts",
        "mean_delta_a",
        "g_plus",
        "g_minus",
        "positive_event_fraction",
        "negative_event_fraction",
    ]
    with (directory / "ngram_position_turn_contributions.csv").open(
        "x", newline=""
    ) as out:
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        for (
            gram,
            pos,
            cohort,
            turn,
            n,
            prompts,
            delta,
            gain,
            loss,
            good,
            bad,
        ) in db.execute(
            """
            SELECT gram,position,cohort,turn,sum(n),count(*),sum(delta),sum(gain),sum(loss),sum(improved),sum(harmed)
            FROM direction_prompts GROUP BY gram,position,cohort,turn ORDER BY gram,position,cohort,turn
        """
        ):
            writer.writerow(
                dict(
                    zip(
                        columns,
                        [
                            gram,
                            pos,
                            cohort,
                            turn,
                            n,
                            prompts,
                            delta / n,
                            gain / n,
                            loss / n,
                            good / n,
                            bad / n,
                        ],
                    )
                )
            )


def write_directions(db, directory, *, min_count, epsilon):
    """Pool positions/turns before classification; prompts never gate a label."""
    write_context_details(db, directory)
    columns = [
        "ngram_token_ids",
        "cohort",
        "classification",
        "classification_zh",
        "events",
        "prompts",
        "improved_events",
        "harmed_events",
        "neutral_events",
        "positive_event_fraction",
        "negative_event_fraction",
        "neutral_event_fraction",
        "mean_delta_a",
        "mean_abs_delta_a",
        "g_plus",
        "g_minus",
        "positive_prompts",
        "negative_prompts",
        "neutral_prompts",
        "positive_prompt_fraction",
        "negative_prompt_fraction",
        "neutral_prompt_fraction",
        "prompt_mean_delta_a",
    ]
    # Pool the same prompt family across positions and MT-Bench turns exactly once.
    query = """SELECT gram,cohort,prompt,sum(n),sum(delta),sum(gain),sum(loss),sum(improved),sum(harmed)
        FROM direction_prompts GROUP BY gram,cohort,prompt ORDER BY gram,cohort,prompt"""
    counts = defaultdict(Counter)
    coverage = defaultdict(
        lambda: dict(
            events=0,
            sufficient_events=0,
            positive_gain=0.0,
            negative_loss=0.0,
            sufficient_positive_gain=0.0,
            sufficient_negative_loss=0.0,
        )
    )
    with (directory / "ngram_directions.csv").open("x", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        for (gram, cohort), rows in groupby(db.execute(query), key=lambda r: r[:2]):
            n = prompts = positive = negative = improved = harmed = 0
            delta = gain = loss = prompt_delta = 0.0
            for _, _, _, events, summed, plus, minus, good, bad in rows:
                n += events
                prompts += 1
                delta += summed
                gain += plus
                loss += minus
                improved += good
                harmed += bad
                mean = summed / events
                positive += mean > epsilon
                negative += mean < -epsilon
                prompt_delta += mean
            row = dict(
                ngram_token_ids=gram,
                cohort=cohort,
                events=n,
                prompts=prompts,
                improved_events=improved,
                harmed_events=harmed,
                neutral_events=n - improved - harmed,
                positive_event_fraction=improved / n,
                negative_event_fraction=harmed / n,
                neutral_event_fraction=(n - improved - harmed) / n,
                mean_delta_a=delta / n,
                mean_abs_delta_a=(gain + loss) / n,
                g_plus=gain / n,
                g_minus=loss / n,
                positive_prompts=positive,
                negative_prompts=negative,
                neutral_prompts=prompts - positive - negative,
                positive_prompt_fraction=positive / prompts,
                negative_prompt_fraction=negative / prompts,
                neutral_prompt_fraction=(prompts - positive - negative) / prompts,
                prompt_mean_delta_a=prompt_delta / prompts,
            )
            label = classify(row, min_count, epsilon)
            row.update(classification=label, classification_zh=LABELS[label])
            writer.writerow(row)
            counts[cohort][label] += 1
            key = (len(json.loads(gram)), cohort)
            cov = coverage[key]
            cov["events"] += n
            cov["positive_gain"] += gain
            cov["negative_loss"] += loss
            if n >= min_count:
                cov["sufficient_events"] += n
                cov["sufficient_positive_gain"] += gain
                cov["sufficient_negative_loss"] += loss
    with (directory / "ngram_direction_summary.csv").open("x", newline="") as out:
        writer = csv.DictWriter(
            out, fieldnames=["cohort", "classification", "classification_zh", "groups"]
        )
        writer.writeheader()
        for cohort, labels in sorted(counts.items()):
            for label in LABELS:
                writer.writerow(
                    dict(
                        cohort=cohort,
                        classification=label,
                        classification_zh=LABELS[label],
                        groups=labels[label],
                    )
                )
    coverage_rows = []
    for (order, cohort), cov in sorted(coverage.items()):
        coverage_rows.append(
            dict(
                ngram_order=order,
                cohort=cohort,
                **cov,
                event_coverage=cov["sufficient_events"] / cov["events"],
                positive_gain_coverage=(
                    cov["sufficient_positive_gain"] / cov["positive_gain"]
                    if cov["positive_gain"]
                    else None
                ),
                negative_loss_coverage=(
                    cov["sufficient_negative_loss"] / cov["negative_loss"]
                    if cov["negative_loss"]
                    else None
                ),
            )
        )
    return dict(
        group_counts={k: dict(v) for k, v in counts.items()},
        coverage_by_order=coverage_rows,
        thresholds=dict(
            min_events=min_count,
            delta_epsilon=epsilon,
            majority_fraction=0.7,
            mixed_min_fraction=0.2,
            mixed_min_events_per_sign=2,
        ),
        note="Descriptive event-based labels, not significance tests. Group = n-gram × cohort, pooled across positions and turns. No prompt-count requirement. Prompt fractions are auxiliary, with MT-Bench turns deduplicated by prompt family. Coverage is separate for each n-gram order; do not add orders. Labels describe the whole Engram increment.",
    )
