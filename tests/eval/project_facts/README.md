# Optional memory extraction evaluation

This frozen six-case corpus covers reported technology, repeated claims,
speculation, a limited negative probe, prompt injection and sixteen memories with
distractors. `corpus-v5.lock.json` pins the input bytes, private expectations and
native experiment template. The original corpus lock and experiment template
remain frozen. Current recipes use `experiment-template.json`, which selects
the reset catalog at `@1`. The recipe selects the common-source `fact` interface
and the keyed Audit interfaces. Upload only the selected `cases/*.json` snapshots.
Keep `corpus.json`, expectations, rubric and authoring files outside the material
provided to a Worker. The dataset's privateChecks retains the private rubric.

Install the opt-in manifests from `configs/evals/project-facts` in a private
operator root based on the shipped catalog, then add their new identities with
`contractor server config lock --root <root>`. They select the existing worker@1
policy and ordinary Audit execution. Each member gets its own execution Project.
The default catalog does not automatically install or run this Eval.

1. Verify the frozen inputs with `python3 tests/eval/project_facts/prepare_recipe.py verify`.
2. Create an evaluation Project (kind evaluation). Upload each selected snapshot with the ordinary Project Artifact API, under
   namespace `fact-eval` and its case ID as name, preserving bytes and application/json.
   Save `{case-id: exact-uploaded-revision}` in a local revisions JSON file.
3. Generate a DatasetInput with `prepare_recipe.py dataset --project <project-id>
   --revisions <file> --output <private-dataset.json>` and import it through
   POST /v1/projects/{projectId}/eval-datasets.
4. Generate a CreateExperiment with `prepare_recipe.py experiment
   --dataset-revision <returned-revision> --output <experiment.json>`, create it
   through the ordinary Eval API, review prepare/readiness, and explicitly start
   its exact plan. `--case technology --repetitions 1` is the deterministic
   integration slice. Full quality work uses repeated samples, three by default.

Both arms initially select the same profile. This is a mechanism baseline, not
an A/B improvement claim. A candidate change needs new immutable configuration
identities and a reviewed experiment. The model receives no expected labels or
additional matching parameters. The native result proves ordinary Audit execution and observed usage. This
profile declares no generated machine report: its facts are reached through the
member Audit, isolated Project and immutable source receipts. The required private
human-review rubric leaves quality incomplete until an authenticated review.
Mechanism acceptance does not count that incomplete review as passing quality.

For every member report:

- unsupported assertion rate: observations unsupported by their original memory,
  divided by all observations; distinguish explicitly qualified inference;
- duplicate/spam rate: redundant or irrelevant observations divided by all
  observations, counting source submissions rather than deduplicated cards;
- useful-fact recall: supported useful observations retained divided by the
  case's private useful expectations; state when this denominator is zero;
- model calls, actual tokens, duration and known cost from ordinary Eval usage;
  unknown cost stays unknown and failed/unrun members remain in denominators.

A scripted Gateway proves execution, publication, isolation and collection only.
It cannot establish semantic model quality. No live experiment is run by tests.

The opt-in relevance candidate uses `project-facts-eval@2`, with a new instruction
path, template and Workflow. The shipped standalone extraction is also available
as `project-facts-extract@2`. Existing `@1` identities remain pinned. Candidate
instructions distinguish reusable target observations from extraction bookkeeping,
retain scoped negative probes, and explain exact source selection, cursor-only
reads and repair attempts under the existing eight-call budget. They contain no
corpus answers. Model policy and execution limits remain the configured baseline.

Verify with `prepare_recipe.py verify --candidate`, then prepare the comparison
with `prepare_recipe.py experiment --candidate --dataset-revision <revision>
--output <private-experiment.json>`. `candidate-v2.lock.json` pins the candidate
configuration and template alongside the unchanged corpus lock. This explicitly
selects `@1` versus `@2`; the default command still prepares the original baseline.
Only instructions and the Audit program's versioned Workflow reference may differ;
models, tools, sampling, execution, source, tasks and scorers must match.
Use debug and verify trace capture in any fresh live comparison. Report repeated
useful-fact recall, spam, tool input failures and tokens per retained useful fact.
No improvement is established until that separate live comparison is reviewed.

The existing native Audit output role findings expects an optional report/machine
Artifact. This recipe does not create that global report or declare it as output.
Reopen portable findings-output collection in the Eval adapter when a concrete
automatic semantic scorer needs a self-contained findings artifact.
