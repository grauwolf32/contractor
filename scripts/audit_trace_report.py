#!/usr/bin/env python3
"""Export a Contractor Audit's Phoenix traces and analyse them using only stdlib."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

AUDIT_LABEL = "contractor.run.label.audit.id"
TASK_LABEL = "contractor.run.label.audit.task"
TOOL = "contractor.worker.tool"
MODEL_NAMES = {"contractor.worker.model", "contractor.planner.model"}
LIFECYCLE = "contractor.worker.a2a_task"
OUTCOMES = {"succeeded", "failed", "cancelled", "rejected", "unavailable"}
BYTE_LIMIT = 16 * 1024 * 1024
NOTICE = (
    "Observed retained Phoenix spans only; not an authoritative Audit total. "
    "Disabled debug, expired/dropped spans, unflushed active Workers and entirely missing "
    "traces cannot be ruled out. Lifecycle counters are comparisons, never extra calls. "
    "Planner dispatch spans are not Worker tool calls. Duration sums include parallel work."
)


class ReportError(Exception):
    """An operator-safe error: never include backend bodies or captured content."""


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON member")
            result[key] = value
        return result

    def invalid_constant(_):
        raise ValueError("non-finite JSON number")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_constant)


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def private_file(path):
    return open(path, "x", encoding="utf-8", opener=lambda p, flags: os.open(p, flags, 0o600))


def write_json(path, value):
    with private_file(path) as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def text_value(value, maximum=256):
    return value[:maximum] if isinstance(value, str) else ""


def number(value):
    if (
        type(value) not in (int, float)
        or value < 0
        or (type(value) is float and not math.isfinite(value))
    ):
        raise ReportError("Invalid nonnegative span measurement")
    return value


def timestamp(value):
    if not isinstance(value, str):
        raise ReportError("Missing span timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ReportError("Invalid span timestamp") from None
    if result.utcoffset() is None:
        raise ReportError("Span timestamp requires a timezone")
    return result.astimezone(timezone.utc)


def content(value):
    """A content decoding problem is a missing diagnostic, not a successful result."""
    if value is None:
        return None, "missing"
    if isinstance(value, str):
        try:
            value = strict_json(value)
        except (ValueError, RecursionError):
            return None, "malformed"
    if isinstance(value, dict) and value.get("truncated") is True:
        return None, "truncated"
    return value, "available"


def compact_span(span):
    if not isinstance(span, dict):
        raise ReportError("Invalid Phoenix span object")
    context, attrs = span.get("context"), span.get("attributes", {})
    if not isinstance(context, dict) or not isinstance(attrs, dict):
        raise ReportError("Invalid Phoenix span context or attributes")
    trace_id, span_id = context.get("trace_id"), context.get("span_id")
    for value, length in ((trace_id, 32), (span_id, 16)):
        if not isinstance(value, str) or re.fullmatch(f"[0-9a-f]{{{length}}}", value) is None:
            raise ReportError("Invalid Phoenix trace/span ID")
    name = span.get("name")
    if not isinstance(name, str) or not name:
        raise ReportError("Missing Phoenix span name")
    start, end = timestamp(span.get("start_time")), timestamp(span.get("end_time"))
    if end < start:
        raise ReportError("Span end precedes its start")
    outcome = attrs.get("outcome")
    status = span.get("status_code")
    if outcome is not None and not isinstance(outcome, str) or not isinstance(status, str):
        raise ReportError("Invalid Phoenix span outcome/status")
    if outcome not in OUTCOMES:
        outcome = {"OK": "succeeded", "ERROR": "failed"}.get(status, "unknown")
    row = {
        "trace_id": trace_id,
        "span_id": span_id,
        "name": name,
        "audit_id": text_value(attrs.get(AUDIT_LABEL)),
        "task": text_value(attrs.get(TASK_LABEL)),
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "duration_ms": number(attrs["duration.ms"])
        if "duration.ms" in attrs
        else (end - start).total_seconds() * 1000,
        "outcome": outcome,
    }
    if name == TOOL:
        output, state = content(attrs.get("contractor.output"))
        error = output.get("error") if isinstance(output, dict) else None
        details = error if isinstance(error, dict) else {}
        row.update(
            {
                "tool": text_value(attrs.get("tool.name")) or "(unavailable)",
                "output_state": state,
                "error_code": text_value(details.get("code"))
                or (text_value(output.get("error_code")) if isinstance(output, dict) else "")
                or text_value(attrs.get("error.code")),
                "error_type": text_value(attrs.get("error.type")),
                "error_message": text_value(details.get("message"), 500)
                or text_value(error, 500)
                or (text_value(output.get("message"), 500) if isinstance(output, dict) else ""),
                "retryable": details.get("retryable")
                if type(details.get("retryable")) is bool
                else None,
            }
        )
    if name in MODEL_NAMES:
        row["model"] = text_value(attrs.get("gen_ai.request.model")) or "(unavailable)"
        for key, attr in (
            ("input_tokens", "gen_ai.usage.input_tokens"),
            ("output_tokens", "gen_ai.usage.output_tokens"),
            ("cached_input_tokens", "gen_ai.usage.cache_read.input_tokens"),
        ):
            value = attrs.get(attr)
            if value is not None and type(value) is not int:
                raise ReportError("Invalid integer token count")
            row[key] = number(value) if value is not None else None
    if name == LIFECYCLE:
        for key in ("tool_calls", "model_calls"):
            value = attrs.get("counts." + key)
            if value is not None and type(value) is not int:
                raise ReportError("Invalid integer lifecycle count")
            row["reported_" + key] = number(value) if value is not None else None
    return row


class Budget:
    def __init__(self, args, clock=time.monotonic):
        self.args, self.clock = args, clock
        self.deadline = clock() + args.deadline
        self.bytes = 0

    def remaining(self):
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise ReportError("Overall deadline exceeded")
        return remaining

    def charge(self, size):
        self.remaining()
        self.bytes += size
        if self.bytes > self.args.max_bytes:
            raise ReportError("Byte budget exceeded; raise --max-bytes for a larger export")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        fp.close()
        raise ReportError("Phoenix redirect refused; use the final --phoenix-url")


class Phoenix:
    def __init__(self, args, budget):
        parsed = urlsplit(args.phoenix_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or (
                parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            )
        ):
            raise ReportError("Use an HTTP(S) Phoenix base URL without credentials/query/fragment")
        self.args, self.budget = args, budget
        self.base = args.phoenix_url.rstrip("/")
        self.opener = build_opener(NoRedirect())
        self.headers = {"Accept": "application/json", "Accept-Encoding": "identity"}
        token = os.environ.get(args.api_key_env)
        if token:
            self.headers["Authorization"] = "Bearer " + token

    def pages(self, filters):
        cursor, seen = None, set()
        while True:
            query = {"limit": self.args.page_size, **filters}
            if cursor:
                query["cursor"] = cursor
            url = (
                self.base
                + "/v1/projects/"
                + quote(self.args.project, safe="")
                + "/spans?"
                + urlencode(query)
            )
            request = Request(url, headers=self.headers)
            try:
                timeout = min(self.args.timeout, self.budget.remaining())
                with self.opener.open(request, timeout=timeout) as response:
                    if response.headers.get("Content-Encoding", "identity") != "identity":
                        raise ReportError("Unexpected compressed Phoenix response")
                    raw = bytearray()
                    while True:
                        self.budget.remaining()
                        part = response.read1(min(65536, BYTE_LIMIT + 1 - len(raw)))
                        self.budget.charge(len(part))
                        raw.extend(part)
                        if len(raw) > BYTE_LIMIT:
                            raise ReportError("Phoenix response exceeds 16 MiB; reduce --page-size")
                        if not part:
                            break
                page = strict_json(raw)
            except HTTPError as error:
                code = error.code
                error.close()
                raise ReportError(
                    f"Phoenix HTTP {code}; check endpoint, project and access"
                ) from None
            except (URLError, OSError, HTTPException, ValueError, RecursionError):
                raise ReportError("Phoenix request failed or returned invalid JSON") from None
            if not isinstance(page, dict) or not isinstance(page.get("data"), list):
                raise ReportError("Invalid Phoenix page")
            if len(page["data"]) > self.args.page_size or "next_cursor" not in page:
                raise ReportError("Invalid Phoenix page bounds/cursor")
            next_cursor = page["next_cursor"]
            if next_cursor is not None and (not isinstance(next_cursor, str) or not next_cursor):
                raise ReportError("Invalid Phoenix cursor")
            if next_cursor is not None and (next_cursor in seen or not page["data"]):
                raise ReportError("Phoenix cursor repeated or empty page cannot advance")
            yield page["data"]
            if next_cursor is None:
                break
            seen.add(next_cursor)
            if len(seen) > self.args.max_spans:
                raise ReportError("Phoenix pagination budget exceeded")
            cursor = next_cursor


class Archive:
    def __init__(self, stream, args, budget):
        self.stream, self.args, self.budget = stream, args, budget
        self.rows, self.traces = {}, set()
        self.duplicates = 0

    def add(self, span):
        self.budget.remaining()
        row = compact_span(span)
        key = (row["trace_id"], row["span_id"])
        if key in self.rows:
            if self.rows[key] != row:
                raise ReportError("Conflicting duplicate span")
            self.duplicates += 1
            return
        if len(self.rows) >= self.args.max_spans:
            raise ReportError("Span budget exceeded; raise --max-spans")
        self.traces.add(row["trace_id"])
        if len(self.traces) > self.args.max_traces:
            raise ReportError("Trace budget exceeded; raise --max-traces")
        line = encode(span) + "\n"
        if len(line.encode("utf-8")) > BYTE_LIMIT:
            raise ReportError("Span JSONL line exceeds 16 MiB")
        self.stream.write(line)
        self.rows[key] = row


def export_phoenix(args, budget, archive):
    client = Phoenix(args, budget)
    # Audit labels are only copied onto Worker lifecycle and Planner invocation spans.
    for page in client.pages({"attribute": AUDIT_LABEL + ":" + encode(args.audit_id)}):
        for span in page:
            if compact_span(span)["audit_id"] != args.audit_id:
                raise ReportError("Phoenix Audit filter returned a foreign span")
            archive.add(span)
    for trace_id in sorted(archive.traces):
        for page in client.pages({"trace_id": trace_id}):
            for span in page:
                if compact_span(span)["trace_id"] != trace_id:
                    raise ReportError("Phoenix trace filter returned a foreign span")
                archive.add(span)


def read_archive(path, archive):
    lines = 0
    with path.open("rb") as stream:
        while True:
            archive.budget.remaining()
            raw = stream.readline(BYTE_LIMIT + 1)
            if not raw:
                return
            archive.budget.charge(len(raw))
            lines += 1
            if len(raw) > BYTE_LIMIT or lines > archive.args.max_spans:
                raise ReportError("Offline JSONL line/count budget exceeded")
            try:
                span = strict_json(raw)
            except (ValueError, RecursionError):
                raise ReportError("Invalid Phoenix JSONL span") from None
            archive.add(span)


def durations(rows):
    values = sorted(row["duration_ms"] for row in rows)
    if not values:
        return {
            "total_ms": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "max_ms": None,
        }
    return {
        "total_ms": round(sum(values), 3),
        "mean_ms": round(sum(values) / len(values), 3),
        "p50_ms": round(values[math.ceil(len(values) * 0.5) - 1], 3),
        "p95_ms": round(values[math.ceil(len(values) * 0.95) - 1], 3),
        "max_ms": round(values[-1], 3),
    }


def call_stats(rows):
    outcomes = Counter(row["outcome"] for row in rows)
    errors = sum(outcomes[key] for key in ("failed", "rejected", "unavailable"))
    return {
        "calls": len(rows),
        "succeeded": outcomes["succeeded"],
        "errors": errors,
        "cancelled": outcomes["cancelled"],
        "unknown": outcomes["unknown"],
        "error_rate_pct": round(100 * errors / len(rows), 3) if rows else None,
        **durations(rows),
    }


def analyse(archive, audit_id):
    rows = list(archive.rows.values())
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["trace_id"]].append(row)
    tools, models, errors, trace_rows, model_rows = [], [], [], [], []
    for trace_id, spans in sorted(grouped.items()):
        labels = {row["audit_id"] for row in spans if row["audit_id"]}
        if labels != {audit_id}:
            raise ReportError("Archive contains an unlabelled or foreign Audit trace")
        task_labels = sorted({row["task"] for row in spans if row["task"]})
        trace_tools = [row for row in spans if row["name"] == TOOL]
        trace_models = [row for row in spans if row["name"] == "contractor.worker.model"]
        lifecycle = [row for row in spans if row["name"] == LIFECYCLE]
        for row in trace_tools:
            row["task"] = ",".join(task_labels)
            if row["outcome"] not in {"succeeded", "unknown"}:
                errors.append(
                    {
                        key: row.get(key)
                        for key in (
                            "trace_id",
                            "span_id",
                            "task",
                            "tool",
                            "outcome",
                            "error_code",
                            "error_type",
                            "error_message",
                            "retryable",
                            "output_state",
                        )
                    }
                )
        tools.extend(trace_tools)
        trace_stats = call_stats(trace_tools)
        models.extend(row for row in spans if row["name"] in MODEL_NAMES)
        expected = {}
        for field in ("tool_calls", "model_calls"):
            values = [row["reported_" + field] for row in lifecycle]
            expected[field] = sum(values) if values and all(v is not None for v in values) else None
        has_worker = bool(lifecycle or trace_tools or trace_models)
        counters_match = (
            all(
                expected[key] == count
                for key, count in (
                    ("tool_calls", len(trace_tools)),
                    ("model_calls", len(trace_models)),
                )
            )
            if has_worker and all(v is not None for v in expected.values())
            else None
        )
        trace_rows.append(
            {
                "trace_id": trace_id,
                "tasks": task_labels,
                "spans": len(spans),
                "worker_trace": has_worker,
                "worker_lifecycle_spans": len(lifecycle),
                "tool_calls": len(trace_tools),
                "tool_errors": trace_stats["errors"],
                "tool_cancelled": trace_stats["cancelled"],
                "worker_model_calls": len(trace_models),
                "reported_tool_calls": expected["tool_calls"],
                "reported_model_calls": expected["model_calls"],
                "lifecycle_counts_match": counters_match,
                "start_time": min(row["start_time"] for row in spans),
                "end_time": max(row["end_time"] for row in spans),
            }
        )
    by_tool = defaultdict(list)
    for row in tools:
        by_tool[row["tool"]].append(row)
    tool_rows = [{"tool": name, **call_stats(values)} for name, values in sorted(by_tool.items())]
    by_model = defaultdict(list)
    for row in models:
        by_model[row["model"]].append(row)
    for name, values in sorted(by_model.items()):
        model_row = {"model": name, **call_stats(values)}
        for key in ("input_tokens", "output_tokens", "cached_input_tokens"):
            known = [row[key] for row in values if row[key] is not None]
            model_row[key] = sum(known) if known else None
            model_row[key + "_measured_calls"] = len(known)
        model_rows.append(model_row)
    error_counts = Counter(
        (r["tool"], r["outcome"], r["error_code"], r["error_type"], r["error_message"])
        for r in errors
    )
    error_groups = [
        {
            "tool": k[0],
            "outcome": k[1],
            "error_code": k[2],
            "error_type": k[3],
            "error_message": k[4],
            "count": v,
        }
        for k, v in sorted(error_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    coverage = Counter(row["output_state"] for row in tools)
    summary = {
        "schema": "contractor.audit-trace-report.v1",
        "audit_id": audit_id,
        "coverage_notice": NOTICE,
        "traces_found": bool(rows),
        "trace_count": len(grouped),
        "span_count": len(rows),
        "observed_start_time": min((r["start_time"] for r in rows), default=None),
        "observed_end_time": max((r["end_time"] for r in rows), default=None),
        "tool_calls": call_stats(tools) if rows else None,
        "tool_output_coverage": dict(sorted(coverage.items())),
        "model_calls": len(models) if rows else None,
        "traces_with_counter_mismatch": sum(
            r["lifecycle_counts_match"] is False for r in trace_rows
        ),
        "worker_traces_without_counter_comparison": sum(
            r["worker_trace"] and r["lifecycle_counts_match"] is None for r in trace_rows
        ),
        "tools": tool_rows,
        "error_groups": error_groups,
        "models": model_rows,
        "traces": trace_rows,
    }
    return summary, tools, errors


def write_csv(path, fields, rows):
    def cell(value):
        if isinstance(value, (list, dict)):
            value = encode(value)
        if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r", "\n")):
            value = "'" + value
        return value

    with private_file(path) as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: cell(row.get(key)) for key in fields})


def md_cell(value):
    if value is None:
        return "—"
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "&#124;")
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("`", "&#96;")
    )


def table(title, fields, rows):
    return [
        "",
        "## " + title,
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
        *["| " + " | ".join(md_cell(row.get(key)) for key in fields) + " |" for row in rows],
    ]


def write_reports(output, summary, tools, errors):
    write_json(output / "summary.json", summary)
    tool_fields = [
        "tool",
        "calls",
        "succeeded",
        "errors",
        "cancelled",
        "unknown",
        "error_rate_pct",
        "total_ms",
        "mean_ms",
        "p50_ms",
        "p95_ms",
        "max_ms",
    ]
    call_fields = [
        "trace_id",
        "span_id",
        "task",
        "tool",
        "outcome",
        "start_time",
        "duration_ms",
        "output_state",
        "error_code",
        "error_type",
    ]
    error_fields = [
        "trace_id",
        "span_id",
        "task",
        "tool",
        "outcome",
        "error_code",
        "error_type",
        "error_message",
        "retryable",
        "output_state",
    ]
    trace_fields = [
        "trace_id",
        "tasks",
        "spans",
        "tool_calls",
        "tool_errors",
        "tool_cancelled",
        "worker_model_calls",
        "reported_tool_calls",
        "reported_model_calls",
        "lifecycle_counts_match",
        "start_time",
        "end_time",
    ]
    for filename, fields, rows in (
        ("tools.csv", tool_fields, summary["tools"]),
        ("tool-calls.csv", call_fields, tools),
        ("errors.csv", error_fields, errors),
        ("traces.csv", trace_fields, summary["traces"]),
    ):
        write_csv(output / filename, fields, rows)
    lines = [
        "# Audit trace report",
        "",
        "Audit: " + md_cell(summary["audit_id"]),
        "",
        NOTICE,
        "",
    ]
    if not summary["traces_found"]:
        lines += [
            (
                "**No labelled traces found.** Check Audit ID, Phoenix project, debug configuration "
                "and retention. Call counts are unavailable."
            )
        ]
    else:
        counts = summary["tool_calls"]
        lines += [
            (
                f"Observed: {summary['trace_count']} traces, {summary['span_count']} spans, "
                f"{counts['calls']} Worker tool calls, {counts['errors']} errors, "
                f"{counts['cancelled']} cancellations, {counts['unknown']} unknown outcomes; "
                f"{summary['model_calls']} model calls."
            ),
            "",
            (
                f"Lifecycle counter mismatches: {summary['traces_with_counter_mismatch']}; "
                f"Worker traces without a counter comparison: "
                f"{summary['worker_traces_without_counter_comparison']}."
            ),
            "",
            "Tool output coverage: " + md_cell(encode(summary["tool_output_coverage"])) + ".",
            (
                "Error rate excludes cancellations and includes failed/rejected/unavailable outcomes. "
                "Percentiles use nearest rank. Missing token measurements remain null; "
                "cached input is a subset of input tokens."
            ),
        ]
        lines += table(
            "Tools (top 50 by calls; all rows in tools.csv)",
            tool_fields,
            sorted(summary["tools"], key=lambda r: (-r["calls"], r["tool"]))[:50],
        )
        lines += table(
            "Error groups (top 50)",
            ["tool", "outcome", "error_code", "error_type", "error_message", "count"],
            summary["error_groups"][:50],
        )
        lines += table(
            "Slowest tool calls (top 20)",
            ["tool", "duration_ms", "outcome", "trace_id", "span_id"],
            sorted(tools, key=lambda r: (-r["duration_ms"], r["span_id"]))[:20],
        )
        lines += table(
            "Models (known usage; top 50)",
            [
                "model",
                "calls",
                "errors",
                "input_tokens",
                "input_tokens_measured_calls",
                "output_tokens",
                "output_tokens_measured_calls",
                "cached_input_tokens",
            ],
            summary["models"][:50],
        )
    lines += [
        "",
        (
            "Full captured span content: spans.jsonl. Error messages: errors.csv. "
            "All aggregate rows and per-trace counter comparisons: summary.json and traces.csv."
        ),
        "",
    ]
    with private_file(output / "report.md") as stream:
        stream.write("\n".join(lines))


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_id", help="existing Contractor Audit ID")
    parser.add_argument("--phoenix-url", default="http://127.0.0.1:6006", help="Phoenix base URL")
    parser.add_argument("--project", default="default", help="Phoenix project name/ID")
    parser.add_argument(
        "--api-key-env",
        default="PHOENIX_API_KEY",
        help="environment variable containing Phoenix bearer key",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="analyse a previously exported Phoenix spans.jsonl offline",
    )
    parser.add_argument("--output", type=Path, help="new output directory (must not exist)")
    parser.add_argument("--page-size", type=positive_int, default=10)
    parser.add_argument("--max-spans", type=positive_int, default=100000)
    parser.add_argument("--max-traces", type=positive_int, default=10000)
    parser.add_argument(
        "--max-bytes",
        type=positive_int,
        default=1024**3,
        help="total HTTP/offline input byte budget",
    )
    parser.add_argument(
        "--timeout", type=positive_int, default=30, help="socket timeout in seconds"
    )
    parser.add_argument(
        "--deadline", type=positive_int, default=900, help="overall budget in seconds"
    )
    args = parser.parse_args(argv)
    # Public ResourceId grammar; Phoenix metadata capture uses the same 256-byte ceiling.
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}", args.audit_id):
        parser.error("invalid Audit ID")
    if args.page_size > 1000:
        parser.error("--page-size must be <= 1000")
    if not args.project or any(c in args.project for c in "/?#"):
        parser.error("invalid Phoenix project identifier")
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output = Path("reports/audit-traces") / (args.audit_id + "-" + stamp)
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    except OSError:
        print(
            "Cannot create output directory; use a new writable --output",
            file=sys.stderr,
        )
        return 1
    manifest = {
        "audit_id": args.audit_id,
        "state": "incomplete",
        "source": "offline" if args.input else "phoenix",
        "coverage_notice": NOTICE,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "project": args.project if not args.input else None,
    }
    manifest_path = args.output / "manifest.json"
    try:
        write_json(manifest_path, manifest)
        budget = Budget(args)
        with private_file(args.output / "spans.jsonl") as stream:
            archive = Archive(stream, args, budget)
            if args.input:
                read_archive(args.input, archive)
            else:
                export_phoenix(args, budget, archive)
        budget.remaining()
        summary, tools, errors = analyse(archive, args.audit_id)
        write_reports(args.output, summary, tools, errors)
        manifest.update(
            state="exported" if summary["traces_found"] else "no_traces",
            span_count=summary["span_count"],
            duplicate_spans_ignored=archive.duplicates,
            input_bytes=budget.bytes,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        # Replace only our own manifest, after all output files have closed successfully.
        replacement = args.output / ".manifest.json"
        write_json(replacement, manifest)
        replacement.replace(manifest_path)
        print("Report: " + str((args.output / "report.md").resolve()))
        if not summary["traces_found"]:
            print("No Audit traces found; tool counts are unavailable", file=sys.stderr)
            return 3
        counts = summary["tool_calls"]
        print(
            f"Observed {counts['calls']} Worker tool calls; {counts['errors']} errors; "
            f"{counts['cancelled']} cancellations. Coverage: retained traces only."
        )
        return 0
    except (ReportError, OSError, ValueError, RecursionError) as error:
        message = (
            str(error) if isinstance(error, ReportError) else "Local I/O or data encoding failed"
        )
        print(message + "; export remains incomplete (see manifest.json)", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "Interrupted; export remains incomplete (see manifest.json)",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    sys.exit(main())
