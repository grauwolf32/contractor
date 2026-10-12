"""Telemetry export settings as the Server sends them.

The Server pins every export value and sends its defaults when a Worker
RuntimeConfig omits the export or retry block (RCF-R102); the Runtime has no
defaults of its own. api/testdata/v1alpha1/runtime-settings-bounds-cases.json
holds those defaults for both languages.
"""

from __future__ import annotations

import json
from pathlib import Path

from contractor_runtime.contracts import TelemetryExportSettings

_CASES = (
    Path(__file__).resolve().parents[3]
    / "api"
    / "testdata"
    / "v1alpha1"
    / "runtime-settings-bounds-cases.json"
)

SERVER_EXPORT = TelemetryExportSettings.model_validate(
    json.loads(_CASES.read_text(encoding="utf-8"))["serverDefaults"]["export"]
)
