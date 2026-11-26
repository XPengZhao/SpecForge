"""Aggregate paired local acceptance contributions without retaining logits."""

import csv
import json
import math
import sqlite3

if __package__:
    from .ngram_directions import write_directions
else:
    from ngram_directions import write_directions


class ContributionSummary:
    def __init__(
        self,
        output_dir,
        *,
        direction_epsilon=0.001,
        direction_min_prompt_events=3,
        direction_min_prompts=5,
    ):
        if not math.isfinite(direction_epsilon) or direction_epsilon < 0:
            raise ValueError("direction epsilon must be finite and nonnegative")
        if min(direction_min_prompt_events, direction_min_prompts) < 1:
            raise ValueError("direction sample thresholds must be positive")
        self.direction_epsilon = direction_epsilon
        self.direction_min_prompt_events = direction_min_prompt_events
        self.direction_min_prompts = direction_min_prompts
        self.output_dir = output_dir
        self.total = self.paired = 0
        self.db = sqlite3.connect(output_dir / "contributions.sqlite")
        self.db.executescript(
            """
            CREATE TABLE direction_prompts (
                gram TEXT, position INTEGER, cohort TEXT, turn INTEGER,
                prompt TEXT, n INTEGER, delta REAL, gain REAL, loss REAL,
                improved INTEGER, harmed INTEGER,
                PRIMARY KEY(gram,position,cohort,turn,prompt));
            CREATE TABLE stats (
                scope TEXT, group_key TEXT, position INTEGER, cohort TEXT,
                n INTEGER, a_on REAL, a_off REAL, gain REAL, loss REAL,
                improved INTEGER, harmed INTEGER,
                PRIMARY KEY(scope,group_key,position,cohort));
            CREATE TABLE prompts (
                scope TEXT, group_key TEXT, position INTEGER, cohort TEXT,
                prompt TEXT, PRIMARY KEY(scope,group_key,position,cohort,prompt));
        """
        )

    def add(self, row):
        self.total += 1
        values = row.get("contribution")
        if values is None:
            return
        self.paired += 1
        on, off = values["a_on"], values["a_off"]
        if not all(math.isfinite(x) and -1e-6 <= x <= 1 + 1e-6 for x in (on, off)):
            raise ValueError("Invalid paired acceptance overlap")
        delta = on - off
        if not math.isclose(values["delta_a"], delta, abs_tol=1e-7):
            raise ValueError("Paired delta disagrees with on/off overlap")
        cohort = "reached" if row["reached"] else "after_rejection"
        groups = [
            ("overall", "", -1),
            ("position", "", row["position"]),
            ("request", row["trace_request_id"], -1),
        ]
        if "turn" in row:
            groups.append(("turn", str(row["turn"]), row["position"]))
        for field in ("ngram2", "ngram3"):
            if row[field]:
                groups.append(("ngram", json.dumps(row[field]), row["position"]))
                self.db.execute(
                    """
                    INSERT INTO direction_prompts VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(gram,position,cohort,turn,prompt) DO UPDATE SET
                    n=n+1, delta=delta+excluded.delta, gain=gain+excluded.gain,
                    loss=loss+excluded.loss, improved=improved+excluded.improved,
                    harmed=harmed+excluded.harmed
                """,
                    (
                        json.dumps(row[field]),
                        row["position"],
                        cohort,
                        row.get("turn", 0),
                        row["prompt_hash"],
                        1,
                        delta,
                        max(delta, 0.0),
                        max(-delta, 0.0),
                        int(delta > self.direction_epsilon),
                        int(delta < -self.direction_epsilon),
                    ),
                )
        for scope, key, position in groups:
            self.db.execute(
                """
                INSERT INTO stats VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(scope,group_key,position,cohort) DO UPDATE SET
                n=n+1, a_on=a_on+excluded.a_on, a_off=a_off+excluded.a_off,
                gain=gain+excluded.gain, loss=loss+excluded.loss,
                improved=improved+excluded.improved, harmed=harmed+excluded.harmed
            """,
                (
                    scope,
                    key,
                    position,
                    cohort,
                    1,
                    on,
                    off,
                    max(delta, 0.0),
                    max(-delta, 0.0),
                    int(delta > 1e-7),
                    int(delta < -1e-7),
                ),
            )
            self.db.execute(
                "INSERT OR IGNORE INTO prompts VALUES (?,?,?,?,?)",
                (scope, key, position, cohort, row["prompt_hash"]),
            )
        if self.paired % 1000 == 0:
            self.db.commit()

    def finish(self, min_count, min_prompts):
        if self.paired and self.paired != self.total:
            self.db.close()
            raise ValueError("Mixed paired/unpaired events; use a fresh trace prefix")
        self.db.commit()
        overall = {}
        columns = [
            "scope",
            "group_key",
            "position",
            "cohort",
            "events",
            "prompts",
            "mean_a_on",
            "mean_a_off",
            "mean_delta_a",
            "g_plus",
            "g_minus",
            "sum_positive_gain",
            "sum_negative_loss",
            "oracle_a",
            "oracle_gain_over_on",
            "improved_events",
            "harmed_events",
            "neutral_events",
        ]
        query = """
            SELECT s.*, p.n_prompts FROM stats s JOIN
            (SELECT scope,group_key,position,cohort,count(*) n_prompts
             FROM prompts GROUP BY scope,group_key,position,cohort) p
            USING(scope,group_key,position,cohort)
            ORDER BY scope,position,group_key,cohort
        """
        with (self.output_dir / "contributions.csv").open("x", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=columns)
            writer.writeheader()
            for (
                scope,
                key,
                pos,
                cohort,
                n,
                on,
                off,
                gain,
                loss,
                improved,
                harmed,
                prompts,
            ) in self.db.execute(query):
                if scope == "ngram" and (n < min_count or prompts < min_prompts):
                    continue
                result = dict(
                    scope=scope,
                    group_key=key,
                    position=pos,
                    cohort=cohort,
                    events=n,
                    prompts=prompts,
                    mean_a_on=on / n,
                    mean_a_off=off / n,
                    mean_delta_a=(on - off) / n,
                    g_plus=gain / n,
                    g_minus=loss / n,
                    sum_positive_gain=gain,
                    sum_negative_loss=loss,
                    oracle_a=(on + loss) / n,
                    oracle_gain_over_on=loss / n,
                    improved_events=improved,
                    harmed_events=harmed,
                    neutral_events=n - improved - harmed,
                )
                writer.writerow(result)
                if scope == "overall":
                    overall[cohort] = result
        directions = write_directions(
            self.db,
            self.output_dir,
            min_count=min_count,
            epsilon=self.direction_epsilon,
        )
        self.db.close()
        return dict(
            ngram_directions=directions,
            enabled=bool(self.paired),
            paired_events=self.paired,
            overall=overall,
            interpretation="Event-weighted local on/off overlap on the ON trajectory; oracle is a binary local switch, not a MAL estimate. N-gram groups attribute the combined Engram increment, not individual tables.",
        )
