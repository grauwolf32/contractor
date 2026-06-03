"""Unit tests for ``tests/eval/trace_vuln_scoring`` — the trace_annotation
task's vuln-finding scorer (family normalization + extraction + matching)."""
from __future__ import annotations

from tests.eval.trace_vuln_scoring import (
    ReportedVuln,
    extract_from_result,
    family_of_class,
    family_of_text,
    score_vulns,
)


def test_family_of_class_groups_to_general_taxonomy():
    assert family_of_class("sqli") == "sqli"
    assert family_of_class("credential-exposure") == "sensitive-data"
    assert family_of_class("pan-storage") == "sensitive-data"
    assert family_of_class("otp-brute-force") == "rate-limit-abuse"
    assert family_of_class("totp-replay") == "auth-crypto"
    assert family_of_class("race-condition") == "business-logic"
    assert family_of_class("totally-unknown") == "other"


def test_family_of_text_keyword_match():
    assert family_of_text("SSRF via unvalidated webhook URL") == "ssrf"
    assert family_of_text("Path Traversal via filename") == "path-traversal"
    assert family_of_text("control_missing: ownership_check") == "idor"
    assert family_of_text("Hardcoded JWT secret in config") == "sensitive-data"
    assert family_of_text("No rate limiting on login endpoint") == "rate-limit-abuse"
    assert family_of_text("TOTP comparison is not constant-time") == "auth-crypto"
    assert family_of_text("just a benign trace summary") is None


def test_extract_from_result_attributes_path_from_trace_target():
    text = (
        "## Annotation blocks\n"
        "# @trace target=POST_/webhooks/test args=req:tainted calls=test_webhook\n"
        "## Security Summary\n"
        "### Finding 1: SSRF via unvalidated webhook test URL\n"
        "- shape: A\n- control_missing: input_validation\n- severity: high\n"
        "### Finding 2: Missing CSRF protection\n"
        "- shape: B\n- control_missing: csrf\n- severity: medium\n"
    )
    out = extract_from_result(text)
    keys = {(r.family, r.path) for r in out}
    assert ("ssrf", "/webhooks/test") in keys
    assert ("csrf", "/webhooks/test") in keys


def test_extract_uses_explicit_method_path_in_block_over_op_path():
    text = (
        "# @trace target=GET_/users/{user_id} args=uid:tainted\n"
        "## Security Summary\n"
        "### Finding: IDOR — note deletion\n"
        "DELETE /notes/{note_id} has no ownership check (idor).\n"
    )
    out = extract_from_result(text)
    # The block names a different endpoint than the operation → block wins.
    # ({param} is collapsed by path normalization.)
    assert ("idor", "/notes/{}") in {(r.family, r.path) for r in out}


def test_score_vulns_exact_path_match_then_lenient():
    expected = [
        {"family": "ssrf", "path": "/webhooks/test", "vuln_class": "ssrf", "severity": "high"},
        {"family": "idor", "path": "/users/{}", "vuln_class": "idor", "severity": "high"},
        {"family": "sqli", "path": "/notes/search", "vuln_class": "sqli", "severity": "high"},
    ]
    reported = [
        ReportedVuln("ssrf", "/webhooks/test"),       # exact
        ReportedVuln("idor", ""),                      # path-agnostic → lenient match
        ReportedVuln("xss", "/somewhere"),             # false positive
    ]
    s = score_vulns(reported, expected)
    assert s.tp == 2 and s.fn == 1 and s.fp == 1
    assert s.recall == round(2 / 3, 3)
    assert s.precision == round(2 / 3, 3)
    assert {m["family"] for m in s.matched} == {"ssrf", "idor"}
    assert s.missed[0]["family"] == "sqli"


def test_score_vulns_no_double_count_same_family():
    # Two expected IDORs, only one reported → exactly one TP, one FN.
    expected = [
        {"family": "idor", "path": "/users/{}", "vuln_class": "idor", "severity": "high"},
        {"family": "idor", "path": "/notes/{}", "vuln_class": "idor", "severity": "high"},
    ]
    reported = [ReportedVuln("idor", "/users/{}")]
    s = score_vulns(reported, expected)
    assert s.tp == 1 and s.fn == 1 and s.fp == 0
    assert s.per_family["idor"] == {"tp": 1, "fn": 1, "fp": 0}
