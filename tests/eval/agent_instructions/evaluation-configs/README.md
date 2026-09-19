# Unpublished instruction evaluation release

This overlay contains twelve new Workflows and two AuditProfiles at exact
`eval-v40-{d1,o1,l1,t1,t2,a1}-{baseline,candidate}@1` selectors. Only A1 has
AuditProfiles. All are unpublished. `manifest.json` pins their bytes and the
original catalog/variant manifests. The source snapshots and default `configs/`
are unchanged.

Materialize `catalog-baseline.json`'s `files` under an isolated config root, then
add `candidate/configs` and this directory's `configs`. Do not combine the overlay
with a moving default catalog or overwrite existing published versions. Before
any future publication, verify the entire resolved dependency closure and reject
selector/content conflicts, including templates, instructions, model policies,
gateways and execution configs. This preparation publishes nothing.

- O1/L1/T1/T2 wrap the exact original or candidate Workflow without changing its
  contract; only the new wrapper identity differs from its source.
- D1 extracts the dependency-discovery stage from the paired OpenAPI workflows.
  Both arms retain hydration, worker template, retries and execution policy;
  both export the discovery result and stop after that stage.
- A1 profiles retain the original `source-checklist@1` inventory, standards,
  inputs/evidence mappings, execution and interaction policies. Each pins its
  own exact child Workflow. The three-item fixture still uses batchSize=2;
  evaluator collection must aggregate the two real child Runs.
- H1/H2 are outside the selected six cases, so no http_explorer wrapper is needed.

Regenerate using the sibling playground's existing Python environment:

```sh
cd ../playground-v2
uv run --project evals python ../contractor/tests/eval/agent_instructions/prepare_configs.py
cd ../contractor
go test ./tests/eval/agent_instructions ./internal/config
```

The Go gate loads the complete frozen catalog plus both overlays and compares
fully resolved contracts. Normalizing only declared identities, instruction
text and dependent template references must leave identical model policy,
tools, Skills, budgets and behavior. This verifies **configuration equality**;
it does not attest the model revision or tools/Skills actually deployed on a
runtime. Those live pins remain unavailable and block strict execution.

See `../README.md` and the paired playground's
`experiments/README-agent-instructions.md` for the frozen recorded control,
read-only demo evidence and remaining live readiness work.
