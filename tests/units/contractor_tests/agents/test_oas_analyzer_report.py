"""Unit tests for oas_analyzer report ordering.

Covers two determinism fixes:

* severity sorting in ``format_vulnerabilities`` used the raw string
  (alphabetical: critical, high, LOW, MEDIUM) — it must use an explicit
  rank map (critical > high > medium > low; unknown severities last);
* ``AnalyticAgent`` iterated a set literal to build its sub-agents, so
  the appsec/datasec/ddos order varied per process — it must be a tuple.
"""
from __future__ import annotations

from contractor.agents.oas_analyzer.sub_agents.analytic_agents import analytic_agent
from contractor.agents.oas_analyzer.sub_agents.report_agent import (
    _severity_rank,
    format_vulnerabilities,
)


def _vuln(severity: str, *, tag: str = "appsec") -> dict:
    return {
        "path": f"/{severity}",
        "method": "get",
        "parameters": ["id"],
        "vulnerability": f"vuln-{severity}",
        "description": f"desc {severity}",
        "severity": severity,
        "confidence": "high",
        "tag": tag,
    }


def test_severity_rank_orders_most_severe_first():
    assert (
        _severity_rank("critical")
        < _severity_rank("high")
        < _severity_rank("medium")
        < _severity_rank("low")
    )


def test_unknown_severity_sorts_last():
    # Pinned: anything outside the known scale goes to the end of the report.
    severities = ["low", "bogus", "critical", "medium", "", "high"]
    ordered = sorted(severities, key=_severity_rank)
    assert ordered == ["critical", "high", "medium", "low", "bogus", ""]


def test_format_vulnerabilities_sorts_by_severity_rank():
    vulnerabilities = [
        _vuln("low"),
        _vuln("critical"),
        _vuln("medium"),
        _vuln("high"),
    ]
    report = format_vulnerabilities(vulnerabilities)
    positions = [report.index(f"vuln-{s}") for s in ("critical", "high", "medium", "low")]
    assert positions == sorted(positions)


def test_analytic_sub_agent_order_is_deterministic():
    # Sub-agents must be built from the ("appsec", "datasec", "ddos") tuple —
    # all appsec bots first, then datasec, then ddos.
    prefixes = [agent.name.split("_")[0] for agent in analytic_agent.sub_agents]
    first_seen = list(dict.fromkeys(prefixes))
    assert first_seen == ["appsec", "datasec", "ddos"]
    # No interleaving: each spec's bots form one contiguous block.
    assert prefixes == sorted(prefixes, key=first_seen.index)
