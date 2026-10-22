from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace

import psutil
import pytest

import contractor_runtime.telemetry.resources as resource_metrics
from contractor_runtime.telemetry.resources import ProcessReading, ResourceCollector, read_process


class FakeTime:
    def __init__(self) -> None:
        self.now = 100.0
        self.waiters: list[tuple[float, asyncio.Future[None]]] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        future = asyncio.get_running_loop().create_future()
        entry = (self.now + delay, future)
        self.waiters.append(entry)
        try:
            await future
        finally:
            self.waiters.remove(entry)

    async def advance(self, seconds: float) -> None:
        self.now += seconds
        for deadline, future in tuple(self.waiters):
            if deadline <= self.now and not future.done():
                future.set_result(None)
        await asyncio.sleep(0)
        await asyncio.sleep(0)


@pytest.mark.parametrize("seconds", [0, 1, 14, 15, 16, 45])
def test_boundaries_periodic_rss_and_cached_final_result(seconds: int) -> None:
    async def scenario() -> None:
        clock = FakeTime()
        calls: list[tuple[float, bool]] = []

        def reader(boundary: bool) -> ProcessReading:
            calls.append((clock.now, boundary))
            return ProcessReading(clock.now * 2, clock.now / 2, int(clock.now) * 1024)

        collector = ResourceCollector(clock=clock.clock, reader=reader, sleep=clock.sleep)
        assert not calls and not clock.waiters
        collector.start()
        collector.start()
        await asyncio.sleep(0)
        for _ in range(seconds):
            await clock.advance(1)
        result = collector.finish()
        assert result.status == "complete"
        assert result.duration_seconds == seconds
        assert result.cpu_user_seconds == seconds * 2
        assert result.cpu_system_seconds == seconds / 2
        assert result.rss_start_bytes == 100 * 1024
        assert result.rss_end_bytes == (100 + seconds) * 1024
        assert result.rss_peak_observed_bytes == result.rss_end_bytes
        assert result.rss_sample_count == 2 + seconds // 15
        assert result.max_sample_gap_seconds == min(seconds, 15)
        assert calls == [(100, True)] + [
            (float(at), False) for at in range(115, 101 + seconds, 15)
        ] + [(100 + seconds, True)]
        encoded = result.model_dump_json(by_alias=True, exclude_none=True)
        await clock.advance(1000)
        assert collector.finish() is result
        assert result.model_dump_json(by_alias=True, exclude_none=True) == encoded
        assert len(calls) == result.rss_sample_count
        assert not clock.waiters
        assert collector._task is not None and collector._task.done()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["exception", "missing", "invalid"])
def test_failed_periodic_rss_keeps_cpu_deltas_and_successful_peak(failure: str) -> None:
    async def scenario() -> None:
        clock = FakeTime()

        def reader(boundary: bool) -> ProcessReading:
            if not boundary:
                if failure == "exception":
                    raise OSError("secret-canary")
                return ProcessReading(rss_bytes=None if failure == "missing" else -1)
            return ProcessReading(clock.now, clock.now / 2, 4096 if clock.now == 100 else 2048)

        collector = ResourceCollector(clock=clock.clock, reader=reader, sleep=clock.sleep)
        collector.start()
        await asyncio.sleep(0)
        await clock.advance(15)
        await clock.advance(5)
        result = collector.finish()
        assert result.status == "partial" and result.reason == "read_failed"
        assert result.cpu_user_seconds == 20
        assert result.cpu_system_seconds == 10
        assert result.rss_sample_count == 2
        assert result.rss_peak_observed_bytes == 4096
        assert result.max_sample_gap_seconds == 20
        assert "secret-canary" not in result.model_dump_json()

    asyncio.run(scenario())


@pytest.mark.parametrize("elapsed", [30, 31, 1000])
def test_skipped_intervals_are_partial_without_catchup_reads(elapsed: int) -> None:
    async def scenario() -> None:
        clock = FakeTime()
        calls = []

        def reader(boundary: bool) -> ProcessReading:
            calls.append(boundary)
            return ProcessReading(clock.now, clock.now, 1)

        collector = ResourceCollector(clock=clock.clock, reader=reader, sleep=clock.sleep)
        collector.start()
        await asyncio.sleep(0)
        await clock.advance(elapsed)
        result = collector.finish()
        assert result.status == "partial" and result.reason == "sampling_gap"
        assert result.max_sample_gap_seconds == elapsed
        assert result.cpu_user_seconds == elapsed
        assert calls == [True, False, True]

    asyncio.run(scenario())


def test_unobserved_scheduled_tick_at_finalization_is_not_complete() -> None:
    async def scenario() -> None:
        clock = FakeTime()
        collector = ResourceCollector(
            clock=clock.clock, reader=lambda _: ProcessReading(0, 0, 0), sleep=clock.sleep
        )
        collector.start()
        clock.now += 20  # Event loop could not service the periodic task.
        result = collector.finish()
        assert result.status == "partial" and result.reason == "sampling_gap"
        assert result.rss_sample_count == 2 and result.rss_peak_observed_bytes == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("boundary", ["start", "end"])
def test_missing_boundary_does_not_fabricate_cpu_or_rss(boundary: str) -> None:
    async def scenario() -> None:
        clock = FakeTime()

        def reader(_: bool) -> ProcessReading:
            if (clock.now == 100) == (boundary == "start"):
                raise OSError
            return ProcessReading(5, 3, 200)

        collector = ResourceCollector(clock=clock.clock, reader=reader)
        collector.start()
        clock.now += 10
        result = collector.finish()
        wire = result.model_dump(by_alias=True, exclude_none=True)
        assert result.status == "partial"
        assert "cpuUserSeconds" not in wire and "cpuSystemSeconds" not in wire
        assert f"rss{boundary.title()}Bytes" not in wire
        assert result.rss_sample_count == 1
        assert result.max_sample_gap_seconds == 10

    asyncio.run(scenario())


def test_counter_reset_and_sequential_collectors_never_reuse_baselines() -> None:
    async def scenario() -> None:
        clock = FakeTime()
        readings = iter(
            [
                ProcessReading(10, 10, 10000),
                ProcessReading(1, 11, 9000),
                ProcessReading(100, 200, 100),
                ProcessReading(102, 203, 200),
            ]
        )
        results = []
        for _ in range(2):
            collector = ResourceCollector(clock=clock.clock, reader=lambda _: next(readings))
            collector.start()
            clock.now += 2
            results.append(collector.finish())
        first, second = results
        assert first.status == "partial" and first.reason == "counter_reset"
        assert first.cpu_user_seconds is None and first.cpu_system_seconds == 1
        assert second.status == "complete"
        assert second.cpu_user_seconds == 2 and second.cpu_system_seconds == 3
        assert second.rss_peak_observed_bytes == 200
        assert second.rss_sample_count == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1, True, 2**54])
def test_invalid_fields_are_omitted_and_valid_cpu_is_preserved(bad: float) -> None:
    async def scenario() -> None:
        collector = ResourceCollector(reader=lambda _: ProcessReading(10, 20, bad))
        collector.start()
        result = collector.finish()
        assert result.status == "partial" and result.reason == "read_failed"
        assert result.cpu_user_seconds == 0 and result.cpu_system_seconds == 0
        assert result.rss_start_bytes is None and result.rss_end_bytes is None
        assert result.rss_peak_observed_bytes is None and result.rss_sample_count == 0

    asyncio.run(scenario())


def test_unsupported_platform_is_unavailable_without_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resource_metrics, "psutil", None)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("unsupported platform must not open a process resource")

    monkeypatch.setattr("builtins.open", forbidden)

    async def scenario() -> None:
        collector = ResourceCollector()
        collector.start()
        result = collector.finish()
        assert result.status == "unavailable" and result.reason == "unsupported_platform"
        assert result.rss_sample_count == 0
        assert result.cpu_user_seconds is None and result.rss_peak_observed_bytes is None

    asyncio.run(scenario())


def test_close_and_owner_cancellation_do_not_invent_recovered_resources() -> None:
    async def scenario() -> None:
        clock = FakeTime()
        calls = []

        def reader(boundary: bool) -> ProcessReading:
            calls.append(boundary)
            return ProcessReading(1, 2, 3)

        collector = ResourceCollector(clock=clock.clock, reader=reader, sleep=clock.sleep)
        collector.start()
        await asyncio.sleep(0)
        collector.close()
        await clock.advance(120)
        assert calls == [True] and not clock.waiters
        result = collector.finish()
        assert result.status == "unavailable"
        assert result.cpu_user_seconds is None and result.rss_end_bytes is None

    asyncio.run(scenario())


def test_process_reader_uses_current_rss_and_only_boundary_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu_calls = []

    def usage() -> SimpleNamespace:
        cpu_calls.append(True)
        return SimpleNamespace(user=2.5, system=1.5, children_user=999, children_system=999)

    current = SimpleNamespace(
        cpu_times=usage,
        memory_info=lambda: SimpleNamespace(rss=12288, peak_rss=999999999, vms=888888888),
    )
    monkeypatch.setattr(psutil, "Process", lambda: current)
    assert read_process(True) == ProcessReading(2.5, 1.5, 12288)
    assert read_process(False) == ProcessReading(rss_bytes=12288)
    assert cpu_calls == [True]


@pytest.mark.parametrize("operation", ["process", "cpu", "rss"])
@pytest.mark.parametrize("failure", ["permission", "os", "unsupported"])
def test_process_reader_preserves_independent_fields_on_failure(
    monkeypatch: pytest.MonkeyPatch, operation: str, failure: str
) -> None:
    error = {
        "permission": psutil.AccessDenied(pid=123),
        "os": OSError("secret-canary"),
        "unsupported": NotImplementedError("secret-canary"),
    }[failure]

    def fail() -> None:
        raise error

    current = SimpleNamespace(
        cpu_times=fail if operation == "cpu" else lambda: SimpleNamespace(user=2.5, system=1.5),
        memory_info=fail if operation == "rss" else lambda: SimpleNamespace(rss=12288),
    )
    monkeypatch.setattr(psutil, "Process", fail if operation == "process" else lambda: current)
    value = read_process(True)
    assert value.reason == ("unsupported_platform" if failure == "unsupported" else "read_failed")
    assert value.cpu_user_seconds == (2.5 if operation == "rss" else None)
    assert value.cpu_system_seconds == (1.5 if operation == "rss" else None)
    assert value.rss_bytes == (12288 if operation == "cpu" else None)
    assert "secret-canary" not in repr(value)
    if operation == "cpu":
        # Periodic observations never call an unavailable CPU reader.
        assert read_process(False) == ProcessReading(rss_bytes=12288)


def test_real_process_reading() -> None:
    value = read_process(True)
    assert value.reason is None
    assert value.cpu_user_seconds is not None and value.cpu_user_seconds >= 0
    assert value.cpu_system_seconds is not None and value.cpu_system_seconds >= 0
    assert value.rss_bytes is not None and value.rss_bytes > 0
    periodic = read_process(False)
    assert periodic.reason is None
    assert periodic.cpu_user_seconds is None and periodic.cpu_system_seconds is None
    assert periodic.rss_bytes is not None and periodic.rss_bytes > 0


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1, True])
def test_bad_monotonic_clock_is_isolated(bad: float) -> None:
    async def scenario() -> None:
        clock = FakeTime()
        collector = ResourceCollector(clock=clock.clock, reader=lambda _: ProcessReading(1, 2, 3))
        collector.start()
        clock.now = bad
        result = collector.finish()
        assert result.status == "unavailable" and result.reason == "read_failed"
        assert result.duration_seconds is None
        await asyncio.sleep(0)
        assert collector._task.done()

    asyncio.run(scenario())


def test_many_samples_keep_only_constant_size_accumulator() -> None:
    async def scenario() -> None:
        clock = FakeTime()
        collector = ResourceCollector(
            clock=clock.clock,
            reader=lambda _: ProcessReading(clock.now, 0, 100),
            sleep=clock.sleep,
        )
        collector.start()
        fields = set(vars(collector))
        await asyncio.sleep(0)
        for _ in range(1000):
            await clock.advance(15)
            assert len(clock.waiters) == 1
        result = collector.finish()
        assert set(vars(collector)) == fields
        assert not any(isinstance(value, (list, dict, set)) for value in vars(collector).values())
        assert result.status == "complete" and result.rss_sample_count == 1002
        assert len(result.model_dump_json(by_alias=True, exclude_none=True)) < 512

    asyncio.run(scenario())
