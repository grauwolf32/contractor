#!/usr/bin/env python3
"""Build pinned ASVS L1 and WSTG source/HTTP review packages from OWASP JSON.

Inputs are local downloads; hashes pin the exact upstream releases. This script
does not fetch, execute tests, or replace previously published package identities.
See docs/guides/audit-standard-sources.md for sources and adaptation policy.
"""

import argparse
import hashlib
import json
from pathlib import Path

ASVS_REVISION = "5cf9b032440be53ce345ab3c130fda46ba1ce7a2"
WSTG_REVISION = "dd33419e10edb22b78d89325a6c2aad9f184e3a2"
ASVS_PACKAGE_VERSION = "5.0.0-l1-source.1"
ASVS_TITLES = {
    "1.2.1": "Contextual response encoding",
    "1.2.2": "Safe URL construction and protocols",
    "1.2.3": "JavaScript and JSON encoding",
    "1.2.4": "Parameterized database queries",
    "1.2.5": "OS command injection prevention",
    "1.3.1": "Untrusted HTML sanitization",
    "1.3.2": "Dynamic code execution",
    "1.5.1": "Restrictive XML parsing",
    "2.1.1": "Documented input validation rules",
    "2.2.1": "Business input validation",
    "2.2.2": "Validation at a trusted service layer",
    "2.3.1": "Business workflow ordering",
    "3.2.1": "Safe browser content interpretation",
    "3.2.2": "Safe text rendering",
    "3.3.1": "Secure cookie attributes and prefixes",
    "3.4.1": "HTTP Strict Transport Security",
    "3.4.2": "CORS origin validation",
    "3.5.1": "Cross-site request forgery protection",
    "3.5.2": "CORS preflight enforcement",
    "3.5.3": "Safe HTTP method semantics",
    "4.1.1": "Correct response content types",
    "4.4.1": "WebSocket transport encryption",
    "5.2.1": "Upload size limits",
    "5.2.2": "Upload extension and content validation",
    "5.3.1": "Non-executable uploaded files",
    "5.3.2": "Safe file path construction",
    "6.1.1": "Documented authentication attack defenses",
    "6.2.1": "Minimum password length",
    "6.2.2": "Password change availability",
    "6.2.3": "Current password verification on change",
    "6.2.4": "Common password rejection",
    "6.2.5": "Unrestricted password composition",
    "6.2.6": "Password input masking",
    "6.2.7": "Password manager and paste support",
    "6.2.8": "Exact password verification",
    "6.3.1": "Credential stuffing and brute-force defenses",
    "6.3.2": "Default account removal",
    "6.4.1": "Secure initial passwords and activation codes",
    "6.4.2": "No password hints or security questions",
    "7.2.1": "Trusted session token verification",
    "7.2.2": "Dynamically generated session tokens",
    "7.2.3": "Cryptographically random session references",
    "7.2.4": "Session rotation on authentication",
    "7.4.1": "Session invalidation on termination",
    "7.4.2": "Session revocation on account removal",
    "8.1.1": "Documented authorization rules",
    "8.2.1": "Function-level authorization",
    "8.2.2": "Object-level authorization",
    "8.3.1": "Authorization at a trusted service layer",
    "9.1.1": "Token signature and MAC verification",
    "9.1.2": "Token algorithm allowlists",
    "9.1.3": "Trusted token verification keys",
    "9.2.1": "Token validity period enforcement",
    "10.4.1": "Exact OAuth redirect URI matching",
    "10.4.2": "Single-use authorization codes",
    "10.4.3": "Short-lived authorization codes",
    "10.4.4": "Client-specific OAuth grant restrictions",
    "10.4.5": "Refresh token replay protection",
    "11.3.1": "Safe cipher modes and padding",
    "11.3.2": "Approved encryption algorithms",
    "11.4.1": "Approved cryptographic hash functions",
    "12.1.1": "Supported TLS protocol versions",
    "12.2.1": "TLS for external HTTP services",
    "12.2.2": "Publicly trusted TLS certificates",
    "13.4.1": "No exposed source control metadata",
    "14.2.1": "No sensitive data in URLs",
    "14.3.1": "Client data cleanup after session termination",
    "15.1.1": "Documented dependency remediation deadlines",
    "15.2.1": "Dependency remediation compliance",
    "15.3.1": "Minimal response data fields",
}
DOCUMENTARY = {"2.1.1", "6.1.1", "8.1.1", "15.1.1"}
DEPLOYMENT = {"12.1.1", "12.2.1", "12.2.2", "13.4.1"}
WSTG_RETIRED = {
    ("WSTG-INFO-09", "Fingerprint Web Application"),
    ("WSTG-INPV-03", "Testing for HTTP Verb Tampering"),
    ("WSTG-INPV-13", "Testing for Buffer Overflow"),
    ("WSTG-ERRH-02", "Testing for Stack Traces"),
}


def pinned_json(path, digest):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Upstream source digest mismatch: {path}")
    return json.loads(data)


def contract(name, *, manual=False, runtime=False):
    assessments = ["blocked", "inconclusive", "not-tested", "violated"]
    if not runtime:
        assessments.append("satisfied")
    return {
        "id": name, "version": "1", "assessments": sorted(assessments),
        "evidenceKinds": ["observation"], "minimumEvidence": 1,
        "maximumEvidence": 8, "humanReview": "required" if manual else "never",
        "rationaleRequired": True,
    }


def package(scheme, version, title, description, source, attribution):
    return {
        "schema": "contractor.audit-standard.v1",
        "standard": {
            "scheme": scheme, "version": version, "title": title,
            "description": description, "source": source,
            "license": {
                "id": "CC-BY-SA-4.0",
                "url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "attribution": attribution, "disclosure": "full",
            },
        },
        "evidenceContracts": [], "entries": [], "mappings": [],
    }


def add_entry(document, identity, title, statement, method, evidence, objective,
              *, level=None, manual=False):
    ref = {"id": evidence, "version": "1"}
    entry = {
        "id": identity, "kind": "requirement", "title": title,
        "statement": statement,
        "applicability": {"mode": "human-review" if manual else "always"},
        "allowedMethods": [method], "evidenceContract": ref,
    }
    if level is not None:
        entry["level"] = level
    document["entries"].append(entry)
    document["mappings"].append({
        "key": identity, "entryIds": [identity], "workflowRole": "check",
        "method": method, "evidenceContract": ref,
        "title": title, "objective": objective,
    })


def build_asvs(upstream):
    if upstream["Version"] != "5.0.0":
        raise ValueError("Expected ASVS 5.0.0")
    document = package(
        "owasp-asvs", ASVS_PACKAGE_VERSION,
        "OWASP ASVS 5.0.0 Level 1 — source and documentation review",
        "All 70 ASVS 5.0.0 Level 1 requirements, reviewed against supplied source, "
        "configuration and documentation. Documentary applicability requires human "
        "review. Deployment-dependent checks cannot be marked satisfied from source "
        "alone. This is a source-review package, not a complete live assessment or certification.",
        {"name": "OWASP Application Security Verification Standard 5.0.0",
         "url": f"https://github.com/OWASP/ASVS/tree/{ASVS_REVISION}/5.0",
         "revision": ASVS_REVISION},
        "OWASP Foundation, OWASP ASVS 5.0.0, CC BY-SA 4.0. All Level 1 requirement "
        "identifiers and statements are reproduced unchanged. Contractor supplies "
        "short titles, source-review mappings and evidence policies as adaptations "
        "under the same license. Package edition: 5.0.0-l1-source.1.",
    )
    document["evidenceContracts"] = [
        contract("bounded-source-verification"),
        contract("documentation-applicability", manual=True),
        contract("deployment-evidence-gap", runtime=True),
    ]
    seen = set()
    for chapter in upstream["Requirements"]:
        for section in chapter["Items"]:
            for item in section["Items"]:
                if str(item["L"]) != "1":
                    continue
                short = item["Shortcode"].removeprefix("V")
                seen.add(short)
                manual = short in DOCUMENTARY
                deployed = short in DEPLOYMENT
                method = "documentation-review" if manual else "configuration-review" if deployed else "source-analysis"
                evidence = "documentation-applicability" if manual else "deployment-evidence-gap" if deployed else "bounded-source-verification"
                objective = (
                    f"Review {ASVS_TITLES[short].lower()} against the exact ASVS "
                    "requirement using the supplied source, configuration and documentation. "
                    "Cite concrete files and relevant data/control paths; preserve missing "
                    "or uninspected surfaces as gaps."
                )
                if deployed:
                    objective += " Deployed behavior cannot be established from source alone; do not report satisfied."
                add_entry(document, "v5.0.0-" + short, ASVS_TITLES[short],
                          item["Description"], method, evidence, objective,
                          level="1", manual=manual)
    if seen != ASVS_TITLES.keys():
        raise ValueError("ASVS Level 1 selection differs from the curated 70 requirements")
    return document


def build_wstg(upstream, *, active=False):
    version = "4.2-http.1" if active else "4.2"
    description = (
        "Active HTTP adaptation of 94 WSTG 4.2 scenarios across 12 categories. "
        "Each check requires approval and an explicit target and authorization scope. "
        "Only directly observed HTTP behavior can be assessed. Browser execution, "
        "raw network/TLS probes, external discovery and isolated multi-user identities "
        "are not provided; affected objectives remain gaps, not passed tests."
        if active else
        "Source-review adaptation of 94 active WSTG 4.2 scenarios across 12 "
        "categories. Three merged aliases and the removed buffer-overflow stub "
        "are excluded. No live requests are executed: source evidence may identify "
        "violations, while live-test objectives remain explicit gaps. This does "
        "not claim a completed dynamic WSTG assessment."
    )
    document = package(
        "owasp-wstg", version,
        "OWASP WSTG 4.2 — " + ("active HTTP review" if active else "source review"),
        description,
        {"name": "OWASP Web Security Testing Guide 4.2",
         "url": f"https://github.com/OWASP/wstg/tree/{WSTG_REVISION}/document/4-Web_Application_Security_Testing",
         "revision": WSTG_REVISION},
        "OWASP Foundation and WSTG contributors, Web Security Testing Guide 4.2, "
        "CC BY-SA 4.0. Scenario titles and objectives are reproduced from the "
        "pinned release checklist. Contractor adds version-qualified identifiers, "
        "versioned links and review mappings/evidence policies under the same license. "
        "Merged aliases and the removed buffer-overflow stub are omitted.",
    )
    evidence = "wstg-http-evidence" if active else "wstg-source-evidence"
    document["evidenceContracts"] = [contract(evidence, runtime=not active)]
    seen = set()
    excluded = set()
    for category in upstream["categories"].values():
        for item in category["tests"]:
            if (item["id"], item["name"]) in WSTG_RETIRED:
                excluded.add((item["id"], item["name"]))
                continue
            identity = item["id"].replace("WSTG-", "WSTG-v42-", 1)
            objectives = [value.strip() for value in item["objectives"] if value.strip()]
            if not objectives or identity in seen:
                raise ValueError(f"Missing objective or duplicate WSTG identifier: {identity}")
            seen.add(identity)
            statement = "\n".join(objectives)
            reference = item["reference"].replace("/stable/", "/v42/")
            objective = (
                "Review the supplied source and configuration for the controls and "
                f"attack surfaces relevant to {identity}. WSTG test objectives:\n"
                f"{statement}\n\n"
                "Trace concrete source evidence and report verified weaknesses as "
                "violated. Do not execute live requests. Preserve every objective "
                "requiring live observation, external discovery or timing as an "
                "explicit gap; use inconclusive, blocked or not-tested when source "
                "evidence is insufficient. Source review cannot establish a passed "
                f"dynamic test. Reference: {reference}"
            )
            if active:
                objective = (
                    f"Perform bounded authorized HTTP checks for {identity}. WSTG test objectives:\n"
                    f"{statement}\n\n"
                    "Use only the declared target and authorization scope, after the "
                    "Server's active-check approval. Preserve request/response evidence "
                    "and control comparisons. Report satisfied only when all applicable "
                    "objectives are established by live evidence in the declared scope. "
                    "Unavailable browser, raw-network/TLS, external-discovery or isolated "
                    "identity capabilities leave explicit gaps. A response code or absence "
                    f"of a demonstrated exploit alone is not a pass. Reference: {reference}"
                )
            add_entry(document, identity, item["name"], statement,
                      "active-test" if active else "source-analysis", evidence, objective)
    if len(seen) != 94 or excluded != WSTG_RETIRED:
        raise ValueError("Unexpected WSTG 4.2 scenario selection")
    return document


def write_package(root, name, document):
    for field in ["entries", "mappings"]:
        document[field].sort(key=lambda item: item.get("id", item.get("key")))
    target = root / "audit-standards" / name / "standard.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    print(f"{target}: {len(document['entries'])} checks")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asvs-source", type=Path, required=True)
    parser.add_argument("--wstg-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("configs"))
    args = parser.parse_args()
    asvs = pinned_json(args.asvs_source, "bcdbec214d70abcfad9284a31d4f9e5134305831d628aad3aa85d7e26626cb35")
    wstg = pinned_json(args.wstg_source, "12e11356b02fc9d6ea1b65554145e14dfdffceb9829ff9d93edf14d3654bdf3c")
    # Validate all documents before writing any package.
    asvs_package = build_asvs(asvs)
    wstg_package, wstg_http_package = build_wstg(wstg), build_wstg(wstg, active=True)
    write_package(args.output_root, "owasp-asvs-5.0.0-l1-source.1", asvs_package)
    write_package(args.output_root, "owasp-wstg-4.2", wstg_package)
    write_package(args.output_root, "owasp-wstg-4.2-http.1", wstg_http_package)


if __name__ == "__main__":
    main()
