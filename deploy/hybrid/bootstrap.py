#!/usr/bin/env python3
"""Idempotent cloud gateway, managed credential and Runtime label bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from common import DEFAULT_STATE, load_state, run, secret, secure_write


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, "redirect refused", headers, fp)


class API:
    def __init__(self, url, token_file):
        parts = urlsplit(url)
        if parts.username or parts.query or parts.fragment:
            raise ValueError("invalid Server URL")
        if parts.scheme != "https" and not (
            parts.scheme == "http" and parts.hostname in {"127.0.0.1", "::1", "localhost"}
        ):
            raise ValueError("HTTP public API is allowed only on loopback")
        self.url = url.rstrip("/")
        self.token = secret(token_file)
        self.opener = build_opener(NoRedirect())

    def request(self, method, path, body=None, headers=None, missing=False):
        values = {"Authorization": "Bearer " + self.token, **(headers or {})}
        raw = None
        if body is not None:
            raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
            values["Content-Type"] = "application/json"
            values["Idempotency-Key"] = "hybrid-" + hashlib.sha256(path.encode() + raw).hexdigest()
        req = Request(self.url + path, data=raw, method=method, headers=values)
        try:
            with self.opener.open(req, timeout=30) as response:
                content = response.read(1024 * 1024 + 1)
                if len(content) > 1024 * 1024:
                    raise RuntimeError("API response exceeded the deployment bound")
                if path == "/readyz" and method == "GET":
                    return None, dict(response.headers)
                return (json.loads(content) if content else None), dict(response.headers)
        except HTTPError as exc:
            exc.close()
            if missing and exc.code == 404:
                return None, {}
            raise RuntimeError(
                f"{method} {path}: HTTP {exc.code}; provider/body details suppressed"
            ) from None

    def pages(self, path):
        cursor = ""
        for _unused in range(100):
            route = path + "?limit=200" + ("&cursor=" + quote(cursor, safe="") if cursor else "")
            page, _ = self.request("GET", route)
            yield from page["items"]
            if not page["page"]["hasMore"]:
                return
            cursor = page["page"].get("nextCursor", "")
            if not cursor:
                raise RuntimeError("missing page cursor")
        raise RuntimeError("deployment page limit exceeded")


def publish(api, state, doc):
    host = next(x for x in doc["hosts"] if x["ssh"] == doc["gateway_host"])
    body = {
        "protocol": "openai-compatible@1",
        "url": f"http://{host['address']}:4000/v1",
        # The existing contract permits HTTP management only on loopback.
        # This origin is encrypted in transit by the dedicated SSH forward.
        "credentialManager": {
            "implementation": "litellm-virtual-keys@1",
            "managementUrl": "http://127.0.0.1:14000",
        },
    }
    path = "/v1/configurations/llm-gateways/cloud-litellm/versions/1"
    resource, _ = api.request("GET", path, missing=True)
    if resource is None:
        resource, _ = api.request(
            "POST",
            "/v1/configurations/llm-gateways",
            {"name": "cloud-litellm", "version": "1", "llmGateway": body},
        )
    if resource["body"] != body:
        raise RuntimeError("cloud-litellm@1 differs; publish a new immutable version")
    ref = resource["ref"]
    gateway_ref = {
        "gatewayId": ref["name"],
        "version": ref["version"],
        "digest": ref["digest"],
    }
    secure_write(state / "gateway-ref.json", json.dumps(gateway_ref, indent=2) + "\n")
    secure_write(
        state / "gateway-admin-bindings.json",
        json.dumps(
            {
                "bindings": [
                    {
                        "llmGateway": gateway_ref,
                        "adminKeyFile": str(state / "secrets/litellm-master-key"),
                    }
                ]
            },
            indent=2,
        )
        + "\n",
    )
    print("cloud-litellm@1 published; exact admin binding prepared")


def selected_model_policies(api):
    settings, _ = api.request("GET", "/v1/operations/settings/models")
    if not settings["configured"]:
        raise ValueError("provide --model-policies when the Server has no compiled model catalog")
    models = {model["id"]: model for model in settings["models"]}
    selection = settings["selection"]
    selectors = []
    for role in ("worker", "summarizer", "audit_completion_worker"):
        model_id = selection["roles"].get(role, selection["defaultModel"])
        selector = models[model_id]["policies"][role]
        if selector not in selectors:
            selectors.append(selector)
    return selectors


def configure(api, state, args):
    routing_path = state / "cloud-routing.json"
    routing = json.loads(routing_path.read_text()) if routing_path.exists() else {}
    credential_id = args.credential_id or routing.get("credentialId", "cloud-workers")
    config_version = args.config_version or routing.get("configVersion", "1")
    gateway = json.loads((state / "gateway-ref.json").read_text())
    policies = []
    for selector in args.model_policies or selected_model_policies(api):
        name, version = selector.rsplit("@", 1)
        resource, _ = api.request(
            "GET",
            "/v1/configurations/model-policies/"
            + quote(name, safe="")
            + "/versions/"
            + quote(version, safe=""),
        )
        ref = resource["ref"]
        policies.append(
            {
                "policyId": ref["name"],
                "version": ref["version"],
                "digest": ref["digest"],
            }
        )
    body = {
        "credentialId": credential_id,
        "llmGateway": gateway,
        "label": "Cloud Runtime Workers",
        "gatewayPolicy": {
            "modelPolicies": policies,
            "maxBudget": args.budget,
            "rpmLimit": 30,
            "maxParallelRequests": 2,
        },
    }
    resource, _ = api.request(
        "GET", "/v1/operations/credentials/" + quote(credential_id, safe=""), missing=True
    )
    if resource is None:
        resource, _ = api.request("POST", "/v1/operations/credentials", body)
    if resource["llmGateway"] != gateway:
        raise RuntimeError("the cloud credential already belongs to another gateway")
    effective = resource["effectivePolicy"]

    def canonical(items):
        return sorted(json.dumps(item, sort_keys=True) for item in items)

    if (
        effective.get("maxBudget") != args.budget
        or canonical(effective["modelPolicies"]) != canonical(policies)
        or effective.get("rpmLimit") != 30
        or effective.get("maxParallelRequests") != 2
    ):
        raise RuntimeError(
            "existing cloud credential policy differs; choose --credential-id and --config-version"
        )
    author = {
        "apiVersion": "contractor/v1alpha1",
        "kind": "RuntimeConfig",
        "metadata": {"name": "cloud-llm", "version": config_version},
        "spec": {
            "worker": {
                "llmGateway": {
                    "gateway": "cloud-litellm@1",
                    "credential": credential_id,
                }
            }
        },
    }
    route = "/v1/operations/runtime-configs/cloud-llm/versions/" + quote(config_version, safe="")
    config, _ = api.request("GET", route, missing=True)
    if config is None:
        config, _ = api.request("POST", "/v1/operations/runtime-configs", author)
    patch = config["document"]["spec"]["worker"]["llmGateway"]
    if patch["gateway"] != gateway or patch["credential"] != credential_id:
        raise RuntimeError("the cloud RuntimeConfig version differs from the requested route")
    binding, _ = api.request("GET", "/v1/operations/runtime-labels/cloud-llm", missing=True)
    if binding is None:
        api.request(
            "PUT",
            "/v1/operations/runtime-labels/cloud-llm",
            {"config": config["ref"]},
            {"If-None-Match": "*"},
        )
    elif binding["config"] != config["ref"]:
        if not args.rebind_label:
            raise RuntimeError("cloud-llm points elsewhere; use --rebind-label for a CAS update")
        api.request(
            "PUT",
            "/v1/operations/runtime-labels/cloud-llm",
            {"config": config["ref"]},
            {"If-Match": '"' + binding["revision"] + '"'},
        )
    secure_write(
        routing_path,
        json.dumps(
            {
                "credentialId": credential_id,
                "configVersion": config_version,
            },
            indent=2,
        )
        + "\n",
    )
    print("Managed cloud credential and cloud-llm Runtime label ready; key retained by Server")


def assign(api, state, doc, args):
    expected = []
    for host in doc["hosts"]:
        for number in range(host["slots"]):
            name = host["ssh"] + "-" + str(number + 1)
            cert = args.pki_root / "agents" / (name + ".crt")
            pem = run(["openssl", "x509", "-in", cert, "-pubkey", "-noout"])
            der = subprocess.run(
                ["openssl", "pkey", "-pubin", "-outform", "DER"],
                input=pem.encode(),
                capture_output=True,
                check=True,
            ).stdout
            expected.append(hashlib.sha256(der).hexdigest())
    principals = {
        p["runtimeAgentId"]: p for p in api.pages("/v1/operations/runtime-agent-principals")
    }
    for identity in expected:
        principal = principals.get(identity)
        if principal is None:
            raise RuntimeError("a cloud Runtime has not registered yet")
        labels = sorted(set(principal["labels"]) | {"cloud-llm"})
        if labels != sorted(principal["labels"]):
            api.request(
                "PUT",
                "/v1/operations/runtime-agent-principals/" + identity + "/labels",
                {"labels": labels},
                {"If-Match": '"' + principal["revision"] + '"'},
            )
    print(f"cloud-llm assigned to {len(expected)} cloud principals; other labels preserved")


def tunnel(state, doc):
    control = state / "gateway-admin.sock"
    existing = subprocess.run(
        ["ssh", "-S", str(control), "-O", "check", doc["gateway_host"]],
        capture_output=True,
        check=False,
    )
    if existing.returncode:
        run(
            [
                "ssh",
                "-M",
                "-S",
                control,
                "-f",
                "-N",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                "ExitOnForwardFailure=yes",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=3",
                "-L",
                "127.0.0.1:14000:172.30.91.2:4000",
                doc["gateway_host"],
            ]
        )
    print("Gateway management forward active on 127.0.0.1:14000 (SSH encrypted)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--token-file", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("publish")
    commands.add_parser("tunnel")
    p = commands.add_parser("configure")
    p.add_argument(
        "--model-policies",
        nargs="+",
        help="Explicit policy selectors; defaults to current Server model choices",
    )
    p.add_argument("--credential-id", help="New immutable credential identity when policy changes")
    p.add_argument("--config-version", help="New immutable cloud-llm version when routing changes")
    p.add_argument("--rebind-label", action="store_true")
    p.add_argument("--budget", type=float, default=5.0)
    p = commands.add_parser("assign")
    p.add_argument("--pki-root", type=Path, required=True)
    args = parser.parse_args()
    state, doc = load_state(args.state)
    if args.command == "tunnel":
        tunnel(state, doc)
        return
    if not args.token_file:
        raise ValueError("--token-file is required")
    api = API(args.server_url, args.token_file)
    if args.command == "publish":
        publish(api, state, doc)
    elif args.command == "configure":
        if not 0 < args.budget <= 100:
            raise ValueError("bootstrap budget must be between 0 and 100")
        configure(api, state, args)
    else:
        assign(api, state, doc, args)


if __name__ == "__main__":
    os.umask(0o077)
    try:
        main()
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"bootstrap: {exc}", file=sys.stderr)
        sys.exit(1)
