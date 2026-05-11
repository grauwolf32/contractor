from unittest.mock import MagicMock

import pytest

from contractor.callbacks.ratelimits import (
    RpmRatelimitCallback,
    TpmRatelimitCallback,
)
from contractor.callbacks.tokens import TokenUsageCallback
from tests.units.contractor_tests.helpers import mk_callback_context


def test_tpm_ratelimit_triggers_sleep__side_effect(monkeypatch):
    # time.time() будет возвращать значения по очереди
    time_mock = MagicMock(side_effect=[1000.0, 1010.0, 1061.0])
    sleep_mock = MagicMock(side_effect=lambda s: None)

    monkeypatch.setattr("time.time", time_mock)
    monkeypatch.setattr("time.sleep", sleep_mock)

    cb = TpmRatelimitCallback(tpm_limit=100, tpm_limit_key="input")

    ctx = mk_callback_context()
    state_key = TokenUsageCallback.global_counter_key()

    # 1) первый вызов — инициализация окна
    ctx.state[state_key] = {"input": 100, "output": 0, "total": 100}
    cb(ctx, MagicMock())

    assert cb.timer_start == 1000.0
    assert cb.token_count == 100
    sleep_mock.assert_not_called()

    # 2) второй вызов — превышение лимита: diff=150, elapsed=10 -> delay=51
    ctx.state[state_key]["input"] = 250
    cb(ctx, MagicMock())

    sleep_mock.assert_called_once()
    assert sleep_mock.call_args.args[0] == pytest.approx(51.0)

    assert len(cb.history) == 1
    assert cb.history[0]["diff"] == 150
    assert cb.history[0]["elapsed_seconds"] == pytest.approx(10.0)

    # после sleep callback стартует новое окно: третий time.time() -> 1061.0
    assert cb.timer_start == 1061.0

    # на всякий случай: time.time() реально дернулся 3 раза
    assert time_mock.call_count == 3


# ─── TpmRatelimitCallback boundaries ──────────────────────────────────────────


class TestTpmBoundaries:
    def test_invalid_limit_key_rejected(self):
        with pytest.raises(AssertionError):
            TpmRatelimitCallback(tpm_limit=100, tpm_limit_key="bogus")

    def test_diff_exactly_at_limit_no_sleep(self, monkeypatch):
        # diff must STRICTLY exceed tpm_limit to trigger a sleep — exactly
        # at the limit is fine.
        monkeypatch.setattr("time.time", MagicMock(side_effect=[1000.0, 1005.0]))
        sleep_mock = MagicMock()
        monkeypatch.setattr("time.sleep", sleep_mock)

        cb = TpmRatelimitCallback(tpm_limit=100, tpm_limit_key="input")
        ctx = mk_callback_context()
        state_key = TokenUsageCallback.global_counter_key()
        ctx.state[state_key] = {"input": 0, "output": 0, "total": 0}

        cb(ctx, MagicMock())  # init
        ctx.state[state_key]["input"] = 100  # diff = 100, exactly at limit
        cb(ctx, MagicMock())

        sleep_mock.assert_not_called()
        assert cb.history == []

    def test_limit_key_output_tracks_output(self, monkeypatch):
        monkeypatch.setattr("time.time", MagicMock(side_effect=[1000.0, 1010.0, 1061.0]))
        sleep_mock = MagicMock()
        monkeypatch.setattr("time.sleep", sleep_mock)

        cb = TpmRatelimitCallback(tpm_limit=50, tpm_limit_key="output")
        ctx = mk_callback_context()
        state_key = TokenUsageCallback.global_counter_key()
        ctx.state[state_key] = {"input": 9999, "output": 0, "total": 0}

        cb(ctx, MagicMock())
        # Bump only output — should trigger; input is intentionally huge to
        # prove the callback isn't looking at it.
        ctx.state[state_key]["output"] = 100
        cb(ctx, MagicMock())

        sleep_mock.assert_called_once()
        assert cb.history[0]["diff"] == 100

    def test_limit_key_total_tracks_total(self, monkeypatch):
        monkeypatch.setattr("time.time", MagicMock(side_effect=[1000.0, 1010.0, 1061.0]))
        sleep_mock = MagicMock()
        monkeypatch.setattr("time.sleep", sleep_mock)

        cb = TpmRatelimitCallback(tpm_limit=50, tpm_limit_key="total")
        ctx = mk_callback_context()
        state_key = TokenUsageCallback.global_counter_key()
        ctx.state[state_key] = {"input": 0, "output": 0, "total": 0}

        cb(ctx, MagicMock())
        ctx.state[state_key]["total"] = 80
        cb(ctx, MagicMock())

        sleep_mock.assert_called_once()


# ─── RpmRatelimitCallback ─────────────────────────────────────────────────────


class TestRpmRatelimit:
    def test_first_call_initializes_window(self, monkeypatch):
        monkeypatch.setattr("time.time", MagicMock(return_value=1000.0))
        sleep_mock = MagicMock()
        monkeypatch.setattr("time.sleep", sleep_mock)

        cb = RpmRatelimitCallback(rpm_limit=3)
        ctx = mk_callback_context()
        cb(ctx, MagicMock())

        assert cb.timer_start == 1000
        assert cb.request_count == 1
        sleep_mock.assert_not_called()

    def test_at_limit_no_sleep(self, monkeypatch):
        # rpm_limit=3 means three requests in the window are fine; only the
        # FOURTH (request_count > limit after increment) triggers a sleep.
        monkeypatch.setattr(
            "time.time",
            MagicMock(side_effect=[1000.0, 1001.0, 1002.0]),
        )
        sleep_mock = MagicMock()
        monkeypatch.setattr("time.sleep", sleep_mock)

        cb = RpmRatelimitCallback(rpm_limit=3)
        ctx = mk_callback_context()
        cb(ctx, MagicMock())  # 1
        cb(ctx, MagicMock())  # 2
        cb(ctx, MagicMock())  # 3 — at limit, no sleep yet

        sleep_mock.assert_not_called()
        assert cb.request_count == 3

    def test_exceeding_limit_triggers_sleep_and_resets(self, monkeypatch):
        # Four time.time() reads: init, 2nd req, 3rd req, 4th req triggers
        # sleep, then a fifth read to set the new window's timer_start.
        monkeypatch.setattr(
            "time.time",
            MagicMock(side_effect=[1000.0, 1001.0, 1002.0, 1003.0, 1064.0]),
        )
        sleep_mock = MagicMock()
        monkeypatch.setattr("time.sleep", sleep_mock)

        cb = RpmRatelimitCallback(rpm_limit=3)
        ctx = mk_callback_context()
        cb(ctx, MagicMock())  # init, request_count=1
        cb(ctx, MagicMock())  # request_count=2
        cb(ctx, MagicMock())  # request_count=3
        cb(ctx, MagicMock())  # request_count=4 > 3 → sleep + reset

        sleep_mock.assert_called_once()
        # delay = 60 - (1003 - 1000) + 1 = 58
        assert sleep_mock.call_args.args[0] == pytest.approx(58.0)
        # Window resets after sleep.
        assert cb.request_count == 1
        assert cb.timer_start == 1064
        assert len(cb.history) == 1
        assert cb.history[0]["elapsed_seconds"] == pytest.approx(3.0)
