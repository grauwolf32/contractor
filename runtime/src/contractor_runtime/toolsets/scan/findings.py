"""Provider-owned problem assertions and selected evidence, without publication intent.

The explicit tool Worker output bindings request durable candidates. Direct model
calls remain observations. Server operations consume only this common contract.
"""

from __future__ import annotations

import hashlib
import json
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from contractor_runtime.toolsets.scan.katana import canonical_url

CANDIDATE_SCHEMA = "contractor.finding.candidates.v1"
EVIDENCE_SCHEMA = "contractor.finding.scanner-evidence.v1"
CANDIDATE_MEDIA_TYPE = "application/vnd.contractor.finding-candidates+json"
EVIDENCE_MEDIA_TYPE = "application/vnd.contractor.finding-evidence+json"
MAX_CANDIDATES = 100
MAX_BATCH_BYTES = 512 << 10
MAX_EVIDENCE_BYTES = 512 << 10
MAX_REPORT_PROJECTION_BYTES = 256 << 10
TECHNIQUES = {
    "boolean-based blind",
    "error-based",
    "inline query",
    "stacked queries",
    "time-based blind",
    "union query",
}
METHOD = re.compile(r"^[A-Z]{1,16}$")
RULE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,159}$")
PARAMETER = re.compile(r"^[^\x00-\x20\x7f]{1,128}$")


def encode(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(value: object) -> str:
    return hashlib.sha256(encode(value)).hexdigest()


def safe_subject(value: str) -> str:
    """Endpoint identity contains parameter names, never query values or userinfo."""
    parsed = urlsplit(canonical_url(value))
    names = sorted({name for name, _ in parse_qsl(parsed.query, keep_blank_values=True)})
    if len(names) > 100 or any(not PARAMETER.fullmatch(name) for name in names):
        raise ValueError("unsupported endpoint parameters")
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode([(n, "") for n in names]), "")
    )


def _text(value: object, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value.encode()) > maximum
        or "\x00" in value
    ):
        raise ValueError("invalid scanner assertion")
    return value


def _method(row: dict) -> str:
    # The actual emitted request, rather than the template default, establishes
    # the method. A missing request cannot manufacture GET or an HTTP capture.
    request = row.get("request")
    if isinstance(request, str):
        first = request.split("\n", 1)[0].strip().split()
        if len(first) == 3 and METHOD.fullmatch(first[0]) and first[2].startswith("HTTP/"):
            return first[0]
    return ""


def _status(row: dict) -> int | None:
    response = row.get("response")
    if isinstance(response, str):
        first = response.split("\n", 1)[0].split()
        if len(first) >= 2 and first[0].startswith("HTTP/") and first[1].isdigit():
            status = int(first[1])
            if 100 <= status <= 599:
                return status
    return None


def nuclei_candidates(observation: dict, *, methods: dict[str, str] | None = None) -> dict:
    rows = observation.get("results", [])
    if not isinstance(rows, list) or len(rows) > MAX_CANDIDATES:
        raise ValueError("invalid bounded Nuclei rows")
    candidates, evidence, seen = [], [], set()
    skipped: dict[str, int] = {}
    surface = []
    surface_omissions = 0
    for row in rows:
        try:
            if not isinstance(row, dict) or not isinstance(row.get("info"), dict):
                raise ValueError
            info = row["info"]
            rule = _text(row.get("template-id"), 160)
            if not RULE.fullmatch(rule):
                raise ValueError
            subject = safe_subject(row.get("matched-at"))
            method, status = _method(row) or (methods or {}).get(rule, ""), _status(row)
            severity = info.get("severity", "unknown")
            # Metadata is independent of problem severity. Missing HTTP facts
            # remain explicit omissions instead of fabricated GET/200 members.
            product = (
                (info.get("metadata") or {}).get("product")
                if isinstance(info.get("metadata", {}), dict)
                else None
            )
            note = ""
            if isinstance(product, str) and re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9 ._+-]{0,63}", product
            ):
                note = "Technology: " + product
                versions = row.get("extracted-results", [])
                if isinstance(versions, list):
                    versions = sorted(
                        {
                            v
                            for v in versions
                            if isinstance(v, str)
                            and re.fullmatch(r"v?\d+(?:\.\d+){1,4}(?:[-+][A-Za-z0-9.-]{1,32})?", v)
                        }
                    )
                    if versions:
                        note += "; reported versions: " + ", ".join(versions[:4])
            metadata = info.get("metadata")
            if isinstance(metadata, dict) and (
                ("product" in metadata and not note)
                or any(key in metadata for key in ("version", "cpe", "technology", "fingerprint"))
            ):
                surface_omissions += 1
            surface.append({"url": subject, "method": method, "status": status, "note": note})
            if severity not in {"low", "medium", "high", "critical"}:
                reason = "informational" if severity == "info" else "severity_unavailable"
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            title = _text(info.get("name"), 512)
            assertion = info.get("description") or title
            assertion = _text(assertion, 8192)
            identity = ["nuclei", rule, subject, method, row.get("matcher-name", "")]
            key = digest(identity)
            if key in seen:
                skipped["duplicate_row"] = skipped.get("duplicate_row", 0) + 1
                continue
            seen.add(key)
            record = {
                "key": key,
                "rule": rule,
                "subject": subject,
                "method": method,
                "severity": severity,
                "matched": True,
                "matcher": str(row.get("matcher-name", ""))[:160],
            }
            for field in (
                "actualPackageDigest",
                "selectedManifestDigest",
                "sourceContentDigest",
                "sourceKind",
                "sourceRef",
                "engineVersion",
                "templateManifest",
                "rulepackArtifact",
                "source",
            ):
                value = observation.get(field)
                if isinstance(value, (str, dict)):
                    record[field] = value
            evidence.append(record)
            package = observation.get("actualPackageDigest") or observation.get(
                "sourceContentDigest", ""
            )
            qualified = (
                ["nuclei", package, rule, subject, method, record["matcher"]]
                if method and package
                else None
            )
            candidates.append(
                {
                    "key": key,
                    "producer": "nuclei",
                    "rule": rule,
                    "title": title,
                    "assertion": assertion,
                    "location": {"url": subject, **({"method": method} if method else {})},
                    "severity": severity,
                    "qualifiedKey": qualified,
                }
            )
        except (ValueError, TypeError, AttributeError, UnicodeError):
            skipped["invalid_row"] = skipped.get("invalid_row", 0) + 1
    return _batch(
        observation, candidates, evidence, skipped, surface, surface_omissions=surface_omissions
    )


def sqlmap_points(output: bytes) -> list[dict]:
    """Retain only actual Parameter/Type blocks, never tested-parameter guesses."""
    current = None
    points: dict[tuple[str, str], set[str]] = {}
    for raw in output.splitlines():
        line = raw.decode("utf-8", errors="replace").strip()
        match = re.fullmatch(
            r"Parameter: (.+) \((GET|POST|Cookie|User-Agent|Referer|Host|URI|\(custom\) POST)\)",
            line,
        )
        if match:
            name, place = match.groups()
            if place == "(custom) POST":
                hinted = re.fullmatch(
                    r"(JSON|JSON-like|XML|SOAP|MULTIPART|ARRAY-like|GRPC-web) (.+)", name
                )
                if hinted:
                    place, name = hinted.groups()
                else:
                    current = None
                    continue
            current = (name, place) if PARAMETER.fullmatch(name) else None
        elif line.startswith("Parameter:"):
            current = None
        elif current is not None and line.startswith("Type:"):
            technique = line[5:].strip().lower()
            if technique in TECHNIQUES:
                points.setdefault(current, set()).add(technique)
    return [
        {"name": name, "place": place, "techniques": sorted(values)}
        for (name, place), values in sorted(points.items())
    ][: MAX_CANDIDATES + 1]


def sqlmap_candidates(
    observation: dict, *, url: str, method: str, points: list[dict], request_digest: str = ""
) -> dict:
    subject = safe_subject(url)
    candidates, evidence, seen = [], [], set()
    for point in points[:MAX_CANDIDATES]:
        name, place, techniques = point["name"], point["place"], point["techniques"]
        identity = ["sqlmap", subject, method, place, name]
        key = digest(identity)
        if key in seen:
            continue
        seen.add(key)
        evidence.append(
            {
                "key": key,
                "subject": subject,
                "method": method,
                "parameter": name,
                "place": place,
                "techniques": techniques,
                "requestDigest": request_digest,
            }
        )
        candidates.append(
            {
                "key": key,
                "producer": "sqlmap",
                "rule": "sql-injection",
                "title": "SQL injection reported by SQLMap",
                "assertion": "SQLMap reported "
                + ", ".join(techniques)
                + " injection in the identified "
                + place
                + " parameter "
                + name
                + ".",
                "location": {"url": subject, **({"method": method} if method else {})},
                "severity": "high",
                "qualifiedKey": identity,
            }
        )
    skipped = {}
    if len(points) > MAX_CANDIDATES:
        skipped["candidate_limit"] = len(points) - MAX_CANDIDATES
    if not points and observation.get("injectionTechniques"):
        skipped["injectable_parameter_unavailable"] = 1
    value = _batch(observation, candidates, evidence, skipped, [])
    if len(points) > MAX_CANDIDATES:
        value["complete"] = False
    return value


def _batch(
    observation: dict,
    candidates: list,
    evidence: list,
    skipped: dict,
    surface: list,
    *,
    surface_omissions: int = 0,
) -> dict:
    complete = observation.get("status") == "completed" and not any(
        observation.get(k)
        for k in (
            "resultsTruncated",
            "outputLimitExceeded",
            "invalidResultLines",
            "stdoutTruncated",
        )
    )
    value = {
        "schema": CANDIDATE_SCHEMA,
        "complete": complete,
        "candidates": candidates,
        "skipped": skipped,
        "evidence": {"schema": EVIDENCE_SCHEMA, "records": evidence},
        "surface": surface,
    }
    if surface_omissions:
        value["surfaceOmissions"] = surface_omissions
    while len(encode(value)) > MAX_REPORT_PROJECTION_BYTES and candidates:
        removed = candidates.pop()
        evidence[:] = [record for record in evidence if record["key"] != removed["key"]]
        skipped["candidate_bytes"] = skipped.get("candidate_bytes", 0) + 1
        value["complete"] = False
    if len(encode(value)) > MAX_REPORT_PROJECTION_BYTES:
        raise ValueError("normalized surface exceeds bound")
    return value
