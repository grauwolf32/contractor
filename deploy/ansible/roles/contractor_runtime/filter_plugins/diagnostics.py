"""Allowlisted deployment diagnostics; never return rejected configuration values."""

import json
import re

NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,62}")
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

GUIDANCE = {
    "Require a Linux host with systemd": "host/platform: Runtime hosts must be Linux with systemd",
    "Require at least the deployed release to be kept": "keep_releases/range: must be at least 1",
    "Require the Control Plane URL": "contractor_runtime_control_plane_url/scheme: requires https",
    "Require well-formed instance entries": "instance/shape: name must match [a-z0-9][a-z0-9_-]{0,62}; state must be present or absent; a present instance needs a port between 1 and 65535",
    "Require at least one present instance": "instances/empty: no present instance",
    "Require unique instance names and ports on the host": "instances/uniqueness: names and ports must be unique",
    "Require applicable capability switches": "capabilities/combination: Known switches are workspace_storage, openapi_validator, likec4_validator, scanners, podman, podman_network; scanners are a subset of nuclei, sqlmap, naabu, ffuf, katana; podman needs local workspace storage; podman_network needs podman",
    "Require an explicit adapter decision and a reachable advertise address": "adapters/advertise_host: adapters require null or a non-empty list; a wildcard listener needs advertise_host",
    "Require values the unit file can carry literally": "unit/literal: empty values, whitespace, quotes, backslashes, $, % and ; are unsupported",
    "Require a log level that keeps the verified events": "log_level/verification: use debug or info, or disable verification",
    "Require a package list for this distribution": "packages/platform: No package groups for this OS family; configure base, podman and podman_network",
    "Require a pinned asset for every needed download": "tools/pin: requires a version and an asset with url and a 64-digit lowercase sha256",
    "Require instance names unique across the play": "instances/uniqueness: names must be unique across hosts",
}


def identity(value, index):
    name = value.get("name") if isinstance(value, dict) else None
    return name if isinstance(name, str) and NAME.fullmatch(name) else f"entry[{index}]"


def input_errors(instances, defaults):
    """Reject unsafe input shapes before Jinja can echo them in type errors."""
    errors = []
    mapping_fields = [
        "extra_environment",
        "capabilities",
        "workspace_limits",
        "podman_limits",
    ]
    sequence_fields = [
        "extra_args",
        "placement_labels",
        "allowed_target_networks",
        "cert_dns",
        "cert_ips",
    ]

    def inspect(value, label):
        for field in mapping_fields:
            if field in value and not isinstance(value[field], dict):
                errors.append(f"{label}: {field}/type requires a mapping")
        for field in sequence_fields:
            if field in value and not isinstance(value[field], (list, tuple)):
                errors.append(f"{label}: {field}/type requires a list")
        caps = value.get("capabilities", {})
        if (
            isinstance(caps, dict)
            and "scanners" in caps
            and not isinstance(caps["scanners"], (list, tuple))
        ):
            errors.append(f"{label}: capabilities.scanners/type requires a list")

    inspect(defaults, "host")
    for field in ["tools", "system_packages"]:
        if not isinstance(defaults[field], dict):
            errors.append(f"host: {field}/type requires a mapping")
    if not isinstance(instances, (list, tuple)):
        return errors + ["instances/type requires a list"]
    for index, entry in enumerate(instances):
        if not isinstance(entry, dict):
            errors.append(f"entry[{index}]/type requires a mapping")
        elif entry.get("state") != "absent":
            inspect(entry, identity(entry, index))
    return errors


def literal_errors(instances, paths, unsafe_pattern):
    unsafe = re.compile(unsafe_pattern)
    errors = []
    for index, instance in enumerate(instances):
        name = identity(instance, index)
        fields = [(f"args[{i}]", value) for i, value in enumerate(instance["args"])]
        for i, (key, value) in enumerate(instance["environment"].items()):
            if not isinstance(key, str) or ENV_NAME.fullmatch(key) is None:
                errors.append(f"{name}: environment[{i}]/environment_name syntax")
            field = (
                f"environment.{key}"
                if isinstance(key, str) and ENV_NAME.fullmatch(key)
                else f"environment[{i}]"
            )
            fields.append((field, str(value)))
        fields.extend((field, str(value)) for field, value in paths.items())
        for field, value in fields:
            if value == "" or unsafe.search(value):
                errors.append(
                    f"{name}: {field}/unit_literal (empty, whitespace or systemd metacharacter)"
                )
    return errors


def resolution_failure(result, task_name, errors, shapes):
    # Failed loop results contain full instances. Only validated identity and
    # fixed field/category text are allowed across this diagnostic boundary.
    try:
        guidance = GUIDANCE.get(
            task_name,
            "configuration/type: resolution failed; check variable types and required fields",
        )
        names = []
        for index, item in enumerate(result.get("results", [])):
            if not item.get("failed"):
                continue
            value = item.get(
                "__contractor_runtime_instance", item.get("__contractor_runtime_entry")
            )
            if value is not None:
                names.append(identity(value, index))
        if task_name == "Require values the unit file can carry literally":
            detail = "; ".join(errors)
        elif task_name == "Require valid configuration structures":
            detail = "; ".join(shapes)
        else:
            detail = guidance
        return (
            "Runtime configuration rejected"
            + (" for " + ", ".join(names) if names else "")
            + ": "
            + detail
        )
    except (AttributeError, TypeError, ValueError):
        return "Runtime configuration rejected: configuration/type; check variable types and required fields"


def registration_summary(stdout, expected):
    summary = {"registered": False, "failed": False, "malformed": False, "missing": []}
    outcomes = {}
    for line in stdout.splitlines():
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
            message = event.get("message", "")
            if not isinstance(message, str):
                raise TypeError("invalid event shape")
            summary["registered"] |= message == "runtime agent registered"
            summary["failed"] |= message.startswith("runtime agent failed")
            ref = event.get("capabilityRef")
            if ref in expected and "probeOutcome" in event:
                # The Runtime can first log available, then reject an invalid
                # tool set. Keep only the latest verdict, never its raw value.
                outcomes[ref] = event["probeOutcome"] == "available"
        except (ValueError, TypeError, AttributeError):
            summary["malformed"] = True
    summary["missing"] = [ref for ref in expected if not outcomes.get(ref, False)]
    return summary


class FilterModule:
    def filters(self):
        return {
            "contractor_literal_errors": literal_errors,
            "contractor_input_errors": input_errors,
            "contractor_resolution_failure": resolution_failure,
            "contractor_registration_summary": registration_summary,
        }
