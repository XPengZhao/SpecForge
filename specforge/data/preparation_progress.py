"""Flushed stage diagnostics and bounded-rate progress for data preparation.

Keep this module dependency-light: pre-tokenized input must not import the
model/data stack merely to report progress.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager


def _report(stage, event, **details):
    fields = " ".join(f"{key}={value!r}" for key, value in details.items())
    print(
        f"[data-prep] {time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"stage={stage} {event} {fields}",
        file=sys.stderr,
        flush=True,
    )


@contextmanager
def preparation_stage(stage: str, **details):
    """Report entry, completion or failure; callers may add final counters."""
    started = time.monotonic()
    _report(stage, "start", **details)
    summary = {}
    try:
        yield summary
    except BaseException as exc:
        _report(
            stage,
            "failed",
            elapsed_s=round(time.monotonic() - started, 3),
            error_type=type(exc).__name__,
            **summary,
        )
        raise
    else:
        _report(
            stage, "done", elapsed_s=round(time.monotonic() - started, 3), **summary
        )


@contextmanager
def preparation_rows(rows, *, total=None):
    """Count rows as they are read, without scanning input to determine length."""
    try:
        from tqdm import tqdm
    except ImportError:  # retain the standard-library-only input fast path
        bar = None
    else:
        bar = tqdm(
            total=total,
            desc="Building prompt tasks",
            unit="samples",
            mininterval=5.0,
            dynamic_ncols=True,
            file=sys.stderr,
        )

    def iterate():
        last_report = time.monotonic()
        for count, row in enumerate(rows, start=1):
            if bar is not None:
                bar.update(1)
            elif time.monotonic() - last_report >= 5.0:
                _report("materialize_tasks", "progress", scanned=count, total=total)
                last_report = time.monotonic()
            yield row

    try:
        yield iterate()
    finally:
        if bar is not None:
            bar.close()
