# Contractor research

This directory contains published working notes, generated reports, and source
surveys. It is tracked with the rest of the documentation. The separate
top-level [`research/`](../../research/README.md) package is the machine-readable
control plane for hypotheses, experiments, and decisions.

## Main reports

- [Interactive research memo](reports/research.html)
- [Research implementation plan](reports/research-implementation-plan.html)
- [Evaluation-runs report](reports/eval-runs-report.html)
- [Quick-wins plan](reports/quickwins-plan.html) and
  [results](reports/quickwins-results.html)
- [Planner report](reports/planner.html)
- [Open-source feature audit](open-source-feature-audit.md)
- [Project review: llama.cpp serving](reports/project-review-llamacpp-serving-2026-06-21.md)

## Supporting material

- [Research additions](research-additions.md)
- [Web sources, July 2026](research-web-sources-2026-07.md)
- [Contractor audits](reports/contractor/)
- [CLI-agent comparisons](reports/cli-agents/)
- [Pentest-agent comparisons](reports/pentest-agents/INDEX.md)
- [Per-direction web research](reports/web/)

The browser-oriented entry point is [`index.html`](index.html). Generated HTML
is committed so the reports remain visible without rerunning their tooling.
Some pentest comparisons intentionally have both Markdown and standalone HTML
forms: the Markdown is review-friendly, while the HTML is the browser report.

## Maintenance

Helper scripts live in [`tools/`](tools/):

```bash
python docs/research/tools/generate-skills-report.py
python docs/research/tools/link-web-research.py
```

Local source clones used to reproduce audits belong in `docs/research/repos/`.
That directory remains ignored; reports and notes outside it are tracked.
