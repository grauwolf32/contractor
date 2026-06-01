"""Regression tests for cli.render._fmt_error (bug M1).

Pre-fix, the field-detection loop always inspected ``result["error"]`` instead
of the candidate field, so tool results carrying their error under
``error_message`` or ``errors`` rendered nothing.
"""

from cli.render import _fmt_error


def _first_line(rendered):
    # Strip ANSI styling noise by checking the human-readable payload only.
    return rendered.strip().splitlines()[0] if rendered else rendered


class TestFmtError:
    def test_error_key(self):
        out = _fmt_error({"error": "boom"})
        assert out is not None and "boom" in out

    def test_error_message_key(self):
        # Pre-fix: returned None (error silently dropped).
        out = _fmt_error({"error_message": "bad input"})
        assert out is not None and "bad input" in out

    def test_errors_key(self):
        out = _fmt_error({"errors": ["a", "b"]})
        assert out is not None and "a" in out and "b" in out

    def test_no_error_returns_none(self):
        assert _fmt_error({"ok": 1, "data": "x"}) is None

    def test_empty_error_values_treated_as_absent(self):
        for empty in ("", [], {}, None):
            assert _fmt_error({"error": empty}) is None

    def test_error_takes_precedence_over_later_fields(self):
        out = _fmt_error({"error": "primary", "error_message": "secondary"})
        assert "primary" in out

    def test_other_fields_still_rendered_after_error(self):
        out = _fmt_error({"error_message": "bad", "code": 42})
        assert "bad" in out and "42" in out
