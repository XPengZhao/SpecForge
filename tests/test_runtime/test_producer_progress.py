"""Publication throughput uses complete wall-time windows, including waits."""

from specforge.launch import _ProducerProgress


def test_throttled_batches_are_counted_and_backpressure_is_included():
    progress = _ProducerProgress(interval=30, now=100)
    assert progress.update(
        now=110, produced=100, in_flight=100, status="producing"
    ) == "throughput=10.0 samples/s in_flight=100 status=producing wait=0%"
    assert progress.update(
        now=120, produced=100, in_flight=200, status="producing"
    ) is None
    progress.wait_seconds += 15
    assert progress.update(
        now=140, produced=0, in_flight=200, status="waiting_consumer"
    ) == "throughput=3.3 samples/s in_flight=200 status=waiting_consumer wait=50%"
    progress.wait_seconds += 30
    assert progress.update(
        now=170, produced=0, in_flight=200, status="waiting_consumer"
    ) == "throughput=0.0 samples/s in_flight=200 status=waiting_consumer wait=100%"
    assert progress.update(
        now=200, produced=300, in_flight=300, status="producing"
    ) == "throughput=10.0 samples/s in_flight=300 status=producing wait=0%"


def test_short_wait_is_visible_after_production_resumes():
    progress = _ProducerProgress(interval=30, now=0)
    progress.update(now=10, produced=100, in_flight=100, status="producing")
    progress.wait_seconds += 6
    assert progress.update(
        now=40, produced=120, in_flight=120, status="producing"
    ) == "throughput=4.0 samples/s in_flight=120 status=producing wait=20%"


def test_disabled_periodic_logging_and_zero_elapsed_time():
    progress = _ProducerProgress(interval=0, now=10)
    assert progress.update(
        now=10, produced=10, in_flight=10, status="producing"
    ) is None
    assert progress.update(
        now=11, produced=0, in_flight=10, status="producing"
    ).startswith("throughput=10.0 samples/s")
    assert progress.update(
        now=100, produced=10, in_flight=20, status="waiting_consumer"
    ) is None
