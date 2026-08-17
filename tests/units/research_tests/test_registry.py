from __future__ import annotations

from pathlib import Path

import pytest

from research.cli import main
from research.registry import RegistryError, load_registry


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _hypothesis(identifier: str, dependencies: list[str] | None = None) -> str:
    direction = "".join(char for char in identifier if char.isalpha())
    dependency_yaml = "\n".join(f"  - {item}" for item in (dependencies or [])) or "  []"
    return f"""\
schema_version: hypothesis/v1
id: {identifier}
direction: {direction}
title: A sufficiently descriptive test hypothesis
claim: This claim is deliberately long enough for strict validation.
mechanism: This mechanism is deliberately long enough for validation.
status: proposed
scenario: offline
primary_metric:
  name: recall
  direction: increase
  minimum_effect: 0.05
dependencies:
{dependency_yaml}
"""


def test_repository_registry_is_valid() -> None:
    registry = load_registry()
    assert "AW1" in registry.hypotheses
    assert "EXP-2026-001" in registry.experiments


def test_unknown_dependency_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path / "hypotheses" / "A1.yaml", _hypothesis("A1", ["Z99"]))
    with pytest.raises(RegistryError, match="unknown dependencies"):
        load_registry(tmp_path)


def test_dependency_cycle_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path / "hypotheses" / "A1.yaml", _hypothesis("A1", ["B1"]))
    _write(tmp_path / "hypotheses" / "B1.yaml", _hypothesis("B1", ["A1"]))
    with pytest.raises(RegistryError, match="dependency cycle"):
        load_registry(tmp_path)


def test_experiment_must_reference_known_hypothesis(tmp_path: Path) -> None:
    source = Path("research/experiments/EXP-2026-001.yaml").read_text(encoding="utf-8")
    _write(tmp_path / "experiments" / "EXP-2026-001.yaml", source)
    with pytest.raises(RegistryError, match="unknown hypothesis"):
        load_registry(tmp_path)


def test_experiment_must_match_complete_metric_contract(tmp_path: Path) -> None:
    hypothesis = Path("research/hypotheses/AW1.yaml").read_text(encoding="utf-8")
    _write(tmp_path / "hypotheses" / "AW1.yaml", hypothesis)
    source = Path("research/experiments/EXP-2026-001.yaml").read_text(encoding="utf-8")
    source = source.replace("minimum_effect: 1.5", "minimum_effect: 2.0", 1)
    _write(tmp_path / "experiments" / "EXP-2026-001.yaml", source)
    with pytest.raises(RegistryError, match="primary metric does not match"):
        load_registry(tmp_path)


def test_cli_validate_and_filter(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate"]) == 0
    assert "1 hypotheses" in capsys.readouterr().out

    assert main(["list", "--direction", "aw"]) == 0
    output = capsys.readouterr().out
    assert "AW1" in output
    assert "Tool-success-conditioned" in output


def test_cli_unknown_hypothesis(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["show", "ZZ999"]) == 1
    assert "unknown hypothesis" in capsys.readouterr().out
