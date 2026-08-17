# Contractor research registry

This directory is the machine-readable control plane for research. Published
reports and working notes live in [`docs/research/`](../docs/research/README.md);
authoritative hypothesis, experiment, and decision records live here.

## Layout

- `hypotheses/` — one falsifiable claim per YAML file.
- `experiments/` — preregistered designs and arm definitions.
- `decisions/` — reviewed outcomes linked to completed experiments.
- `models.py` — strict versioned schemas.
- `registry.py` — loading, uniqueness, dependency, and cross-record checks.
- `cli.py` — validation and inspection.

Production modules under `contractor/` must never import this package. Research
arms should resolve into existing production configuration/factory interfaces.

## Commands

```bash
poetry run python -m research.cli validate
poetry run python -m research.cli list
poetry run python -m research.cli list --direction AW --status ready
poetry run python -m research.cli show AW1
```

After the next `poetry install`, the equivalent `contractor-research` console
command declared in `pyproject.toml` is also available.

## Record workflow

1. Add the hypothesis as `proposed`.
2. Add dependencies, owner, metric, minimum useful effect, and guardrails.
3. Move to `ready` only when instrumentation and fixtures exist.
4. Create an experiment as `draft`; run CPU preflight only.
5. Freeze its fixture IDs, seeds, scorer, exclusions, thresholds, and cost cap.
6. Set `frozen_at` and status `frozen` before the first model call.
7. Store raw run output under `eval_runs/`; do not put generated output here.
8. Add a decision after applying the preregistered rule.
9. Promote only after a separate confirmatory experiment.

The initial AW1 record is a template, not a claim that the other 227 memo
hypotheses have already been normalized. Migrate them direction by direction,
reviewing metrics and dependencies rather than mechanically copying prose.
