"""run_context must honour the module's never-raises contract: a broken
Langfuse client degrades to a no-op span (with a warning), never a crash."""

import sys
import types

import pytest

import contractor.utils.observability as obs


class _WorkingSpanCM:
    """Minimal stand-in for langfuse's start_as_current_span() CM."""

    def __init__(self):
        self.span = object()
        self.entered = False
        self.exited_with: type[BaseException] | None | str = "never-exited"

    def __enter__(self):
        self.entered = True
        return self.span

    def __exit__(self, exc_type, exc, tb):
        self.exited_with = exc_type
        return False


class _BrokenEnterSpanCM(_WorkingSpanCM):
    def __enter__(self):
        raise RuntimeError("span enter failed")


class _BrokenExitSpanCM(_WorkingSpanCM):
    def __exit__(self, exc_type, exc, tb):
        raise RuntimeError("span exit failed")


class _FakeClient:
    def __init__(self, span_cm=None, span_factory_error=None):
        self.span_cm = span_cm
        self.span_factory_error = span_factory_error
        self.flush_count = 0

    def start_as_current_span(self, *, name):
        if self.span_factory_error is not None:
            raise self.span_factory_error
        return self.span_cm

    def update_current_trace(self, **kwargs):
        pass

    def flush(self):
        self.flush_count += 1


def _enable_with_fake_langfuse(monkeypatch, client):
    """Force-enable observability and route `from langfuse import get_client`
    to a fake module — works whether or not langfuse is installed."""
    mod = types.ModuleType("langfuse")
    mod.get_client = lambda: client
    monkeypatch.setitem(sys.modules, "langfuse", mod)
    monkeypatch.setattr(obs, "_enabled", lambda: True)


def test_disabled_yields_none(monkeypatch):
    # Langfuse off → pure no-op (forced off so a dev's .env can't flip it).
    monkeypatch.setattr(obs, "_enabled", lambda: False)
    with obs.run_context(name="run") as span:
        assert span is None


def test_broken_get_client_degrades_to_noop(monkeypatch):
    mod = types.ModuleType("langfuse")

    def _boom():
        raise RuntimeError("no langfuse server")

    mod.get_client = _boom
    monkeypatch.setitem(sys.modules, "langfuse", mod)
    monkeypatch.setattr(obs, "_enabled", lambda: True)

    with obs.run_context(name="run") as span:
        assert span is None


def test_broken_span_factory_degrades_to_noop(monkeypatch):
    client = _FakeClient(span_factory_error=RuntimeError("otel exploded"))
    _enable_with_fake_langfuse(monkeypatch, client)

    body_ran = False
    with obs.run_context(name="run") as span:
        body_ran = True
        assert span is None

    assert body_ran
    assert client.flush_count == 1  # flush still runs on the degraded path


def test_broken_span_enter_degrades_to_noop(monkeypatch):
    client = _FakeClient(span_cm=_BrokenEnterSpanCM())
    _enable_with_fake_langfuse(monkeypatch, client)

    with obs.run_context(name="run") as span:
        assert span is None


def test_broken_span_exit_does_not_raise(monkeypatch):
    cm = _BrokenExitSpanCM()
    client = _FakeClient(span_cm=cm)
    _enable_with_fake_langfuse(monkeypatch, client)

    with obs.run_context(name="run") as span:
        assert span is cm.span  # the working enter still yields the real span

    assert client.flush_count == 1


def test_working_span_closes_and_body_exception_propagates(monkeypatch):
    cm = _WorkingSpanCM()
    client = _FakeClient(span_cm=cm)
    _enable_with_fake_langfuse(monkeypatch, client)

    with (
        pytest.raises(ValueError, match="from the body"),
        obs.run_context(name="run"),
    ):
        raise ValueError("from the body")

    # The span was closed with the in-flight exception (status recorded).
    assert cm.exited_with is ValueError
    assert client.flush_count == 1
