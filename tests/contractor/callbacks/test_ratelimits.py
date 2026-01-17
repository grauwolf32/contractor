import pytest
from unittest.mock import MagicMock

from contractor.callbacks.ratelimits import TpmRatelimitCallback
from contractor.callbacks.tokens import TokenUsageCallback
from tests.contractor.callbacks.helpers import mk_callback_context


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
