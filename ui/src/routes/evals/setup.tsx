import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useId, useState, type ReactNode } from "react";
import { Link, useNavigate, useSearchParams } from "react-router";
import { usePublicAPI } from "../../api/context";
import {
  saveEvalDraft,
  type EvalDraft,
  type EvalExperiment,
} from "../../api/evals";
import { createProject } from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import { useDocumentTitle } from "../../app/document-title";
import { TechnicalDetails } from "../../ui";
import { EvalError, EvalField, EvalFrame } from "./common";
import { AssessmentSetup } from "./assessment-setup";
import { DatasetAuthor } from "./datasets";
import {
  KeyValueValidityProvider,
  useKeyValueValidity,
} from "./key-value-validity";
import { executionKindLabel, expectedMembersText, pinLabel } from "./labels";
import {
  useEvalCapabilities,
  useEvalCases,
  useEvalDatasets,
  useEvalOwner,
  useEvalProjects,
} from "./queries";
import { EvalReadiness } from "./readiness";
import { recoverableMutation } from "./recovery";
import {
  COMPARISON_PURPOSES,
  draftProblem,
  initialEvalDraft,
  PIN_DIMENSIONS,
} from "./setup-model";
import { VariantEditor } from "./variants";

const STEPS = [
  "Variants",
  "Cases and inputs",
  "Assessment and repetitions",
  "Readiness",
] as const;

export function EvalSetupForm({
  experiment,
  onDirtyChange,
  actions,
}: {
  experiment?: EvalExperiment;
  onDirtyChange?: (dirty: boolean) => void;
  actions?: ReactNode;
}) {
  const api = usePublicAPI(),
    owner = useEvalOwner(),
    cache = useQueryClient(),
    navigate = useNavigate();
  const [params] = useSearchParams();
  const authorPanel = useId();
  const [projectId, setProjectId] = useState(
    experiment?.projectId ?? params.get("project") ?? "",
  );
  const [name, setName] = useState(experiment?.name ?? "");
  const [draft, setDraft] = useState<EvalDraft>(() =>
    experiment?.draft
      ? structuredClone(experiment.draft)
      : {
          ...initialEvalDraft(),
          dataset: {
            id: params.get("dataset") ?? "",
            revision: params.get("revision") ?? "",
          },
        },
  );
  const [step, setStep] = useState(0);
  const [author, setAuthor] = useState(false);
  const [workspaceName, setWorkspaceName] = useState("");
  const [dirty, setDirty] = useState(false);
  const [validation, setValidation] = useState<Error | null>(null);
  const keyValues = useKeyValueValidity();
  const projects = useEvalProjects(),
    capabilities = useEvalCapabilities(),
    datasets = useEvalDatasets(projectId),
    cases = useEvalCases(projectId, draft.dataset.id, draft.dataset.revision);
  function update(next: EvalDraft) {
    setDraft(next);
    setDirty(true);
    onDirtyChange?.(true);
    setValidation(null);
  }
  const createWorkspace = useMutation({
    mutationFn: async () => {
      const body = { kind: "evaluation" as const, name: workspaceName };
      return recoverableMutation(owner, "workspace", body, (idempotencyKey) =>
        createProject(api, { request: body, idempotencyKey }),
      );
    },
    onSuccess: async (result) => {
      await cache.invalidateQueries({ queryKey: queryKeys.evals.projects });
      setProjectId(result.projectId);
      setWorkspaceName("");
    },
  });
  const save = useMutation({
    mutationFn: async () => {
      if (keyValues.invalid)
        throw new Error(
          "Resolve duplicate or empty parameter names before saving.",
        );
      const problem = draftProblem(name, draft);
      if (problem) throw new Error(problem);
      const body = { name: name.trim(), draft };
      const current = experiment
        ? { id: experiment.experimentId, revision: experiment.revision }
        : undefined;
      const request = { body, current };
      const operation = `draft:${projectId}`;
      return recoverableMutation(owner, operation, request, (key) =>
        saveEvalDraft(api, projectId, body, key, current),
      );
    },
    onSuccess: async (result) => {
      setDirty(false);
      onDirtyChange?.(false);
      await cache.invalidateQueries({ queryKey: queryKeys.evals.all });
      void navigate(
        `/evals/experiments/${encodeURIComponent(result.experimentId)}/setup`,
      );
    },
  });
  const kind = draft.variants[0]?.kind ?? "workflow";
  const expected = draft.caseIds.length * 2 * draft.repetitions;
  const last = STEPS.length - 1;
  return (
    <div className="eval-setup">
      <nav className="eval-steps" aria-label="Experiment setup steps">
        <ol role="list">
          {STEPS.map((label, index) => (
            <li key={label}>
              <button
                type="button"
                aria-current={step === index ? "step" : undefined}
                disabled={keyValues.invalid && step !== index}
                onClick={() => setStep(index)}
              >
                <span className="eval-step-number" aria-hidden="true">
                  {index + 1}
                </span>
                <span>
                  <span className="ui-visually-hidden">{index + 1}.</span>{" "}
                  {label}
                </span>
              </button>
            </li>
          ))}
        </ol>
      </nav>
      <p className="eval-matrix" aria-live="polite">
        {draft.caseIds.length} cases × 2 variants × {draft.repetitions}{" "}
        repetitions = <strong>{expectedMembersText(expected)}</strong>
      </p>
      <section className="eval-panel eval-setup-panel">
        <fieldset className="eval-setup-step" disabled={save.isPending}>
          {step === 0 ? (
            <>
              <div className="eval-step-heading">
                <h2>Variants</h2>
                <p className="eval-muted">
                  Choose a baseline and a candidate to compare on the same
                  cases.
                </p>
              </div>
              <div className="eval-grid">
                <EvalField label="Experiment name">
                  <input
                    value={name}
                    maxLength={256}
                    onChange={(e) => {
                      setName(e.target.value);
                      setDirty(true);
                      onDirtyChange?.(true);
                      setValidation(null);
                    }}
                  />
                </EvalField>
                <EvalField label="Evaluation workspace">
                  <select
                    value={projectId}
                    disabled={!!experiment}
                    onChange={(e) => {
                      setProjectId(e.target.value);
                      update({
                        ...draft,
                        dataset: { id: "", revision: "" },
                        caseIds: [],
                      });
                    }}
                  >
                    <option value="">Choose a workspace</option>
                    {projects.data
                      ?.filter(
                        (p) =>
                          p.lifecycle === "active" || p.projectId === projectId,
                      )
                      .map((p) => (
                        <option key={p.projectId} value={p.projectId}>
                          {p.name}
                        </option>
                      ))}
                  </select>
                </EvalField>
              </div>
              {!experiment ? (
                <details className="eval-disclosure">
                  <summary>Create an evaluation workspace</summary>
                  <div className="eval-disclosure-body">
                    <EvalField label="New workspace name">
                      <input
                        value={workspaceName}
                        onChange={(e) => setWorkspaceName(e.target.value)}
                      />
                    </EvalField>
                    <div className="eval-actions">
                      <button
                        type="button"
                        className="ui-btn"
                        data-size="sm"
                        disabled={
                          !workspaceName.trim() || createWorkspace.isPending
                        }
                        onClick={() => createWorkspace.mutate()}
                      >
                        Create workspace
                      </button>
                    </div>
                    <EvalError error={createWorkspace.error} />
                  </div>
                </details>
              ) : null}
              <div className="eval-grid">
                <EvalField label="Execution kind">
                  <select
                    value={kind}
                    onChange={(e) =>
                      update({
                        ...draft,
                        variants: draft.variants.map((v) => ({
                          ...v,
                          kind: e.target.value as typeof kind,
                          selector: "",
                          executionConfig: {},
                        })),
                      })
                    }
                  >
                    <option value="workflow">
                      {executionKindLabel("workflow")}
                    </option>
                    <option value="audit">{executionKindLabel("audit")}</option>
                  </select>
                </EvalField>
                <EvalField label="Comparison purpose">
                  <select
                    defaultValue=""
                    onChange={(e) => {
                      const purpose =
                        COMPARISON_PURPOSES[
                          e.target.value as keyof typeof COMPARISON_PURPOSES
                        ];
                      if (purpose)
                        update({
                          ...draft,
                          comparison: {
                            ...draft.comparison,
                            requiredEqual: [...purpose.equal],
                            allowedDifferences: [...purpose.different],
                          },
                        });
                    }}
                  >
                    <option value="">Choose or retain current policy</option>
                    {Object.entries(COMPARISON_PURPOSES).map(
                      ([key, preset]) => (
                        <option key={key} value={key}>
                          {preset.label}
                        </option>
                      ),
                    )}
                  </select>
                </EvalField>
              </div>
              <div className="eval-arms">
                <KeyValueValidityProvider value={keyValues.register}>
                  {[draft.comparison.baseline, draft.comparison.candidate]
                    .map((id) =>
                      draft.variants.find((variant) => variant.id === id)!,
                    )
                    .map((variant) => (
                      <VariantEditor
                        key={variant.id}
                        label={
                          variant.id === draft.comparison.baseline ? "A" : "B"
                        }
                        value={variant}
                        capabilities={capabilities.data}
                        onChange={(next) =>
                          update({
                            ...draft,
                            variants: draft.variants.map((v) =>
                              v.id === variant.id ? next : v,
                            ),
                          })
                        }
                      />
                    ))}
                </KeyValueValidityProvider>
              </div>
              <details className="eval-disclosure">
                <summary>Equality policy</summary>
                <div className="eval-disclosure-body">
                  <p className="eval-muted">
                    Unknown required-equal dimensions block preparation.
                  </p>
                  <div className="eval-policy-grid">
                    {PIN_DIMENSIONS.map((pin) => (
                      <EvalField label={pinLabel(pin)} key={pin}>
                        <select
                          value={
                            draft.comparison.requiredEqual.includes(pin)
                              ? "equal"
                              : draft.comparison.allowedDifferences.includes(
                                    pin,
                                  )
                                ? "different"
                                : "observe"
                          }
                          onChange={(e) =>
                            update({
                              ...draft,
                              comparison: {
                                ...draft.comparison,
                                requiredEqual: [
                                  ...draft.comparison.requiredEqual.filter(
                                    (x) => x !== pin,
                                  ),
                                  ...(e.target.value === "equal" ? [pin] : []),
                                ],
                                allowedDifferences: [
                                  ...draft.comparison.allowedDifferences.filter(
                                    (x) => x !== pin,
                                  ),
                                  ...(e.target.value === "different"
                                    ? [pin]
                                    : []),
                                ],
                              },
                            })
                          }
                        >
                          <option value="equal">Required equal</option>
                          <option value="different">Difference allowed</option>
                          <option value="observe">
                            Observe without a gate
                          </option>
                        </select>
                      </EvalField>
                    ))}
                  </div>
                </div>
              </details>
            </>
          ) : null}
          {step === 1 ? (
            <>
              <div className="eval-step-heading">
                <h2>Cases and inputs</h2>
              </div>
              {!projectId ? (
                <p className="eval-muted">
                  Select an evaluation workspace in Variants first.
                </p>
              ) : (
                <>
                  <div className="eval-inline-fields">
                    <EvalField label="Dataset revision">
                      <select
                        value={draft.dataset.revision}
                        onChange={(e) => {
                          const dataset = datasets.data?.find(
                            (d) => d.revision === e.target.value,
                          );
                          update({
                            ...draft,
                            dataset: {
                              id: dataset?.datasetId ?? "",
                              revision: dataset?.revision ?? "",
                            },
                            caseIds: [],
                          });
                        }}
                      >
                        <option value="">Choose a revision</option>
                        {datasets.data?.map((d) => (
                          <option key={d.revision} value={d.revision}>
                            {d.name} · {d.revision} · {d.caseCount} cases
                          </option>
                        ))}
                      </select>
                    </EvalField>
                    <button
                      type="button"
                      className="ui-btn"
                      aria-expanded={author}
                      aria-controls={author ? authorPanel : undefined}
                      onClick={() => setAuthor(!author)}
                    >
                      Create or import dataset
                    </button>
                  </div>
                  {author ? (
                    <div id={authorPanel}>
                      <DatasetAuthor
                        projectId={projectId}
                        onSaved={(dataset) => {
                          update({
                            ...draft,
                            dataset: {
                              id: dataset.datasetId,
                              revision: dataset.revision,
                            },
                            caseIds: [],
                          });
                          setAuthor(false);
                        }}
                      />
                    </div>
                  ) : null}
                  {cases.isFetching ? (
                    <p role="status">Loading visible cases…</p>
                  ) : null}
                  {cases.data ? (
                    <>
                      <div className="eval-actions">
                        <button
                          type="button"
                          className="ui-btn"
                          data-size="sm"
                          onClick={() =>
                            update({
                              ...draft,
                              caseIds: cases.data!.map((c) => c.id),
                            })
                          }
                        >
                          Select all {cases.data.length} cases
                        </button>
                        <button
                          type="button"
                          className="ui-btn"
                          data-size="sm"
                          onClick={() => update({ ...draft, caseIds: [] })}
                        >
                          Clear selection
                        </button>
                        <span className="eval-muted">
                          {draft.caseIds.length} of {cases.data.length} selected
                        </span>
                      </div>
                      <ul className="eval-case-list" role="list">
                        {cases.data.map((c) => (
                          <li key={c.id}>
                            <label className="eval-checkbox">
                              <input
                                type="checkbox"
                                checked={draft.caseIds.includes(c.id)}
                                onChange={(e) =>
                                  update({
                                    ...draft,
                                    caseIds: e.target.checked
                                      ? [...draft.caseIds, c.id]
                                      : draft.caseIds.filter(
                                          (id) => id !== c.id,
                                        ),
                                  })
                                }
                              />
                              <span>
                                <strong className="eval-mono">{c.id}</strong> ·{" "}
                                {c.task.objective}
                              </span>
                            </label>
                            <small>
                              Inputs:{" "}
                              {Object.keys(c.inputs).join(", ") || "none"};
                              required capabilities:{" "}
                              {c.requires.join(", ") || "none"}
                            </small>
                          </li>
                        ))}
                      </ul>
                    </>
                  ) : null}
                </>
              )}
            </>
          ) : null}
          {step === 2 ? (
            <AssessmentSetup
              draft={draft}
              onChange={update}
              capabilities={capabilities.data}
            />
          ) : null}
          {step === 3 ? <EvalReadiness draft={draft} /> : null}
        </fieldset>
        <EvalError
          error={
            validation ??
            save.error ??
            projects.error ??
            capabilities.error ??
            datasets.error ??
            cases.error
          }
          reload={
            save.error
              ? () => {
                  void cache.invalidateQueries({
                    queryKey: queryKeys.evals.experiment(
                      experiment?.experimentId,
                    ),
                  });
                }
              : undefined
          }
        />
        <footer className="eval-setup-footer">
          <div className="eval-setup-actions">
            <div className="eval-actions">
              <button
                type="button"
                className="ui-btn"
                disabled={step === 0 || keyValues.invalid}
                onClick={() => setStep(step - 1)}
              >
                Back
              </button>
              {step < last ? (
                <button
                  type="button"
                  className="ui-btn"
                  data-variant={experiment ? undefined : "primary"}
                  disabled={keyValues.invalid}
                  onClick={() => setStep(step + 1)}
                >
                  Next step
                </button>
              ) : null}
              <button
                type="button"
                className="ui-btn"
                data-variant={
                  (experiment ? dirty : step === last) ? "primary" : undefined
                }
                disabled={
                  !projectId ||
                  save.isPending ||
                  keyValues.invalid ||
                  (!!experiment && !dirty)
                }
                onClick={() => {
                  const problem = draftProblem(name, draft);
                  if (problem) {
                    setValidation(new Error(problem));
                    return;
                  }
                  save.mutate();
                }}
              >
                {save.isPending ? "Saving…" : "Save draft"}
              </button>
            </div>
            <p className="eval-save-status" role="status">
              {experiment && !dirty
                ? "Saved on server"
                : "Unsaved changes · save this draft before leaving."}
            </p>
          </div>
          {experiment ? (
            <TechnicalDetails summary="Draft details">
              <dl className="eval-facts">
                <dt>Saved revision</dt>
                <dd>{experiment.revision}</dd>
                <dt>Portable experiment ID</dt>
                <dd>
                  <code>{experiment.portableExperimentId}</code>
                </dd>
              </dl>
            </TechnicalDetails>
          ) : null}
          {actions}
          <p className="eval-muted eval-setup-note">
            {experiment && dirty
              ? "Save your changes before preparing this draft. "
              : ""}
            Prepare checks compatibility. Start runs the experiment after you
            confirm.
          </p>
        </footer>
      </section>
    </div>
  );
}

export function EvalNewRoute() {
  useDocumentTitle("New experiment");
  return (
    <EvalFrame
      title="New experiment"
      breadcrumb={[
        { label: "Experiments", to: "/evals" },
        { label: "New experiment" },
      ]}
      description="Compare two versions on a shared dataset. Save, prepare, then start when ready."
      action={
        <Link className="ui-btn" data-size="sm" to="/evals/datasets">
          Manage datasets
        </Link>
      }
    >
      <EvalSetupForm />
    </EvalFrame>
  );
}
