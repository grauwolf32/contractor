# Hypotheses produced by Workflows

## Purpose and scope

This document owns hypothesis production inside an ordinary configured Audit
Workflow: selected grounds, local proposal output, successful-Run publication
and continuation. [Common Audit documents](common-contract.md#ACT-R16) own Task
and result semantics. [Workflow material sets](../configuration/workflows.md#WFL-R96)
and [ArtifactStore](../artifacts/artifact-store.md#ART-R143) own exact input
selection, authorization, binding and retention. [Findings](../findings/findings.md)
own problem records and their reader snapshots.

## Model

A hypothesis is an assertion with scope and explanatory grounds. Its optional
`sources` are exact retained Artifact descriptors. Its `origin` identifies the
producer; sources and origin answer different questions. A proposal contains
producer-local aliases and optional checking work. Saving it creates an ordinary
Run output, with no immediate registry mutation.

## Requirements

**HG-R1.** Any allowed ordinary Workflow may produce hypotheses. Workflow identity,
Stage/Agent names, input names and physical output names do not confer this role.
There is no generator entity, execution mode or additional controller.

**HG-R2.** The pinned Program maps one logical `proposals` role to a required JSON
Workflow output, and one `prior-results` role to a required JSON context input.
The actual Stage graph must expose that context and one unambiguous required
output target from its Worker. The proposal tool requires these roles; context
without an output role is invalid. A binding selects at most one semantic intake
output (`proposals` or `work`). The ordinary Run materializer freezes resolved
roles in `audit-context/hypothesis-proposals`; its Run ID, exact context ref and
versionless Worker target are protocol metadata, not authored authority.

**HG-R3.** Research inputs are declared by Workflow and selected by the caller.
They may be single exact artifacts or native material sets under WFL-R96 and
ART-R143. Input names are unrestricted. The Run-local proposal context names
actual readable exact members and preserves original input identities for
cumulative accounting; duplicate source descriptors share a handle. It grants
no access to refs merely mentioned inside source content. Initial generation
requires no previous Task, result or Finding.

**HG-R4.** A declared `task-results` input is an optional native material set of
new selected settled Task results. Its accompanying observations retain exact
Task IDs, full checked hypothesis snapshots, and technical gaps. Several Tasks
may reference one shared exact result without merging their snapshots or charging
the material more than once. Supported,
refuted and inconclusive results all remain useful materials. The continuation
cursor accounts for new work; it does not define all research inputs. If this
input role is absent, Task history still supplies bounded trigger accounting,
without adding that history to the Workflow's research context.

**HG-R5.** Workflow may select permitted exact materials during its Run. Project
Findings discovery and explicit revision selection use the existing reader and
selection Artifact contract. A changed revision is an explicit conflict; no
implicit latest substitution occurs. A selected snapshot is an ordinary own-Run
artifact and may be used as a ground or an optional checking input. No new live
reader, selector registry or collector is introduced.

**HG-R6.** Proposed hypotheses contain `statement`, `scope`, `grounds` and optional
`sources` naming available material handles. `selected_materials` maps additional
local handles to exact refs in the producer Run. Server derives scope, content
metadata and retained Audit refs. Empty/absent sources are valid; source labels
are not provenance. Persisted revision sources are full exact Audit descriptors,
separate from automatic origin. Editing authored content preserves its sources.

**HG-R7.** Model supplies bounded judgement and local aliases. Runtime validates
content and assembles the typed proposal through `submit_hypothesis_proposals`.
It resolves configured roles, checks exact selections and uses ordinary Artifact
CAS to correct local output. Repeated calls retain prior selected refs; a handle
cannot be rebound to another source. Tool success means a proposal was saved,
not a hypothesis published. Model never assigns persisted IDs/revisions/origin.

**HG-R8.** Only a successful producer Run permits semantic intake. The common
accepting owner validates the complete batch, assigns deterministic hypothesis
IDs, first revisions, origin and optional checking associations, and commits
hypotheses, Tasks, source links and progress with the existing Task receipt.
Hypotheses-only, allowed work-only and explicit empty decisions are valid.
Malformed/missing output is invalid. Failed/cancelled Run diagnostics may be
collected as ordinary materials but publish no hypotheses or checking Tasks.

**HG-R9.** Input handles and additional own-Run refs use one accepting path.
Server resolves exact metadata and authority again before accepting any domain
membership. Foreign scope, hidden material, missing/versionless refs and
incompatible inputs reject the whole batch. System context and proposal output
cannot be selected as research grounds. Input refs selected explicitly must
match the trusted research inputs. Embedded JSON refs never grant access.

**HG-R10.** Optional checking work names an allowed Workflow, objective, optional
local or pinned hypothesis alias, and permitted input handles. Server binds the
full exact assertion automatically. System inputs cannot be overwritten. A
producer response cannot invoke another proposal producer; further work remains
ordinary accepted Tasks dispatched by existing RunService and Scheduler.

**HG-R11.** All original inputs and explicit in-Run selections share LIM-ART-14.
Distinct descriptors count cumulatively across sets, singletons, aliases and
proposal corrections. Complete hypothesis documents and combined hypothesis/Task
mutations independently obey LIM-ACT-1/2; system context obeys LIM-ACT-3. Source
payloads remain ordinary artifacts, without repackaging or recursive expansion.
Accepted grounds use existing Audit artifact links and evidence-byte accounting;
shared retained revisions are charged once.
Project-to-Audit retention preserves original scoped exact identity through
ordinary Artifact lineage. A source selected both in a set and as a singleton
uses that same identity for continuation planning, Run preparation and proposal
input sources; independent sources with equal content remain distinct.
The shared Artifact operation checks initial and additional own-Run selections
against that budget. Context builders add domain fields to its prepared exact
descriptors; they do not re-read sets, infer origins from local names or rebuild
the selected-material budget.

**HG-R12.** A hypothesis is not a Finding. A checking result records its scoped
verdict, tested conditions, limitations and exact hypothesis revision. Refuted
or inconclusive results neither create a Finding nor resolve an existing one.
Hypothesis identity belongs to Audit; Project Findings retain their own owner.
Explicit material publication does not create a second live hypothesis registry.

**HG-R13.** At the drained idle barrier, optional configured continuation precedes
normal Audit completion. It accepts one ordinary Task with a stable source
boundary and selected observation IDs, using existing receipts and claims.
Source queries use bounded indexed seeks, never hydrate whole Task history, and
leave unselected tails available. Trusted continuation Task origin excludes its
own outputs regardless of Workflow/output alias. Initial/manual proposal
producers are also excluded by their pinned semantic role or proposal tool.
Unchanged ticks and valid empty output cause no additional invocation. Available
material capacity bounds a selected prefix; zero capacity closes explicitly
with `material-budget-exhausted`. Task/Run exhaustion likewise has an explicit
stop reason. An explicit owner decision may schedule later ordinary work.

**HG-R14.** Invalid configuration fails before Start. Invalid source/proposal or
permanent admission failure yields a classified permanent outcome, no partial
membership, and an explicit collection gap or continuation stop. This includes
the accepting service's own invalid-request errors. Storage conflicts retain
existing bounded retries and stable identity. Ambiguous acknowledgement is
reconciled through the existing receipt before reading producer context/sources;
source disappearance after commit does not regenerate or duplicate work.
Timeouts retain their owning operation/Run/model deadline.

**HG-R15.** Pause prohibits new dispatch, retains accepted continuation and lets
admitted work finish collection. Resume uses the same Task/Run identities and
exact inputs. Cancel/deletion bars semantic intake under the existing Project/
Audit owner fence, while allowing ordinary output collection. Accepted grounds
are held by ordinary frozen Audit bindings and survive producer Run deletion.
Audit purge removes its registry and holds. Explicit Project publications and
Project Finding evidence keep their existing independent retention.
Common Task admission retains every exact selected set member before accepting
the Task, as well as its JSON descriptor. Run creation is not the first hold:
source history maintenance during Pause must preserve the accepted selection.

**HG-R16.** Projections distinguish saved proposal, accepted hypothesis, planned
check, scoped result and Finding. Accepted hypotheses expose origin, grounds and
exact source descriptors; Tasks/results expose the exact checked revision.
Rejected intake exposes a gap and budget exhaustion a reason, without new
mutable intermediate entity states. Methodology belongs in instructions/Skills;
Server checks structure and authority, not truth or semantic deduplication.

## Interfaces

The common [Audit JSON schema](../../../api/audit/v1/contract.schema.json) owns
`proposals`, `proposal_context`, `proposal_binding` and `hypothesis` shapes.
The [public API](../../../api/openapi/contractor-public-v1.yaml) exposes accepted
hypothesis revisions and ordinary material publication. Sources use the common
[material descriptor](../../../api/v1alpha1/material-set.schema.json).

## Limits, failures and errors

[Limits](../platform/limits.md) LIM-ART-14, LIM-ACT-1/2/3/7 and the pinned Audit
budgets apply together. [Failure model](../platform/failure-model.md) and
[error registry](../platform/errors.md) own stable classifications, including
`audit_document_invalid`, invalid input and `storage_transaction_conflict`.

## Boundary decisions

Ordinary exact bindings avoid copying selected payloads into a new archive;
reconsider packaging only for a concrete offline/export consumer. Publication
after Run success keeps cancellation and atomic membership under the existing
owner; reconsider immediate writes only for a concrete dependency inside Run.
Explicit grounds plus separate origin preserve verifiable material links without
a knowledge graph. Existing readers cover in-Run selections; a universal reader,
new registry, receipt, lease or controller has no caller here.
