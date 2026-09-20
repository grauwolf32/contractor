import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";
import { Link, useNavigate, useSearchParams } from "react-router";
import { usePublicAPI } from "../../api/context";
import {
  saveEvalDraft,
  type EvalDraft,
  type EvalExperiment,
} from "../../api/evals";
import { createProject } from "../../api/projects";
import { EvalError, EvalField, EvalFrame } from "./common";
import {
  useEvalCapabilities,
  useEvalCases,
  useEvalDatasets,
  useEvalOwner,
  useEvalProjects,
} from "./queries";
import {
  COMPARISON_PURPOSES,
  draftProblem,
  initialEvalDraft,
  PIN_DIMENSIONS,
} from "./setup-model";
import { VariantEditor } from "./variants";
import { DatasetAuthor } from "./datasets";
import { AssessmentSetup } from "./assessment-setup";
import { EvalReadiness } from "./readiness";
import { finishMutation, mutationKey } from "./recovery";

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
      const result = await createProject(api, {
        request: body,
        idempotencyKey: await mutationKey(owner, "workspace", body),
      });
      await finishMutation(owner, "workspace", body);
      return result;
    },
    onSuccess: async (result) => {
      await cache.invalidateQueries({ queryKey: ["evals", "projects"] });
      setProjectId(result.projectId);
      setWorkspaceName("");
    },
  });
  const save = useMutation({
    mutationFn: async () => {
      const problem = draftProblem(name, draft);
      if (problem) throw new Error(problem);
      const body = { name: name.trim(), draft };
      const current = experiment
        ? { id: experiment.experimentId, revision: experiment.revision }
        : undefined;
      const request = { body, current };
      const operation = `draft:${projectId}`;
      const result = await saveEvalDraft(
        api,
        projectId,
        body,
        await mutationKey(owner, operation, request),
        current,
      );
      await finishMutation(owner, operation, request);
      return result;
    },
    onSuccess: async (result) => {
      setDirty(false);
      onDirtyChange?.(false);
      await cache.invalidateQueries({ queryKey: ["evals"] });
      void navigate(
        `/evals/experiments/${encodeURIComponent(result.experimentId)}/setup`,
      );
    },
  });
  const kind = draft.variants[0]?.kind ?? "workflow";
  const expected = draft.caseIds.length * 2 * draft.repetitions;
  return (
    <div className="eval-setup">
      <nav className="eval-steps" aria-label="Experiment setup steps">
        {STEPS.map((label, index) => (
          <button
            type="button"
            key={label}
            className="secondary-button"
            aria-current={step === index ? "step" : undefined}
            onClick={() => setStep(index)}
          >
            <span className="eval-step-number">{index + 1}.</span>{" "}
            <span>{label}</span>
          </button>
        ))}
      </nav>
      <p className="eval-matrix" aria-live="polite">
        {draft.caseIds.length} cases × 2 variants × {draft.repetitions}{" "}
        repetitions = <strong>{expected} expected members</strong>
      </p>
      <section className="panel eval-panel">
        <fieldset disabled={save.isPending}>
          {step === 0 ? (
            <>
              <h2>Variants</h2>
              <p className="eval-section-description">
                Choose a baseline and a candidate to compare on the same cases.
              </p>
              <div className="form-grid">
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
                <details>
                  <summary>Create an evaluation workspace</summary>
                  <EvalField label="New workspace name">
                    <input
                      value={workspaceName}
                      onChange={(e) => setWorkspaceName(e.target.value)}
                    />
                  </EvalField>
                  <button
                    type="button"
                    className="secondary-button"
                    disabled={
                      !workspaceName.trim() || createWorkspace.isPending
                    }
                    onClick={() => createWorkspace.mutate()}
                  >
                    Create workspace
                  </button>
                  <EvalError error={createWorkspace.error} />
                </details>
              ) : null}
              <div className="form-grid">
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
                    <option value="workflow">Workflow</option>
                    <option value="audit">Audit</option>
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
              <div className="eval-variants">
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
              </div>
              <details>
                <summary>Equality policy</summary>
                <p>Unknown required-equal dimensions block preparation.</p>
                {PIN_DIMENSIONS.map((pin) => (
                  <EvalField label={pin} key={pin}>
                    <select
                      value={
                        draft.comparison.requiredEqual.includes(pin)
                          ? "equal"
                          : draft.comparison.allowedDifferences.includes(pin)
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
                              ...(e.target.value === "different" ? [pin] : []),
                            ],
                          },
                        })
                      }
                    >
                      <option value="equal">Required equal</option>
                      <option value="different">Difference allowed</option>
                      <option value="observe">Observe without a gate</option>
                    </select>
                  </EvalField>
                ))}
              </details>
            </>
          ) : null}
          {step === 1 ? (
            <>
              <h2>Cases and inputs</h2>
              {!projectId ? (
                <p>Select an evaluation workspace in Variants first.</p>
              ) : (
                <>
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
                      <option value="">Choose an immutable revision</option>
                      {datasets.data?.map((d) => (
                        <option key={d.revision} value={d.revision}>
                          {d.name} · {d.revision} · {d.caseCount} cases
                        </option>
                      ))}
                    </select>
                  </EvalField>
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => setAuthor(!author)}
                  >
                    Create or import dataset
                  </button>
                  {author ? (
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
                  ) : null}
                  {cases.isFetching ? (
                    <p role="status">Loading visible cases…</p>
                  ) : null}
                  {cases.data ? (
                    <>
                      <div className="eval-actions">
                        <button
                          type="button"
                          className="secondary-button"
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
                          className="secondary-button"
                          onClick={() => update({ ...draft, caseIds: [] })}
                        >
                          Clear selection
                        </button>
                      </div>
                      <ul className="eval-case-list">
                        {cases.data.map((c) => (
                          <li key={c.id}>
                            <label>
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
                              <strong>{c.id}</strong> · {c.task.objective}
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
                    queryKey: ["evals", "experiment", experiment?.experimentId],
                  });
                }
              : undefined
          }
        />
        <footer className="eval-setup-footer">
          <p className="eval-save-status" role="status">
            {experiment && !dirty
              ? `Saved on server · revision ${experiment.revision}`
              : "Unsaved changes · save this draft before leaving."}
          </p>
          <div className="eval-setup-actions">
            <div className="eval-actions">
              <button
                type="button"
                className="secondary-button"
                disabled={step === 0}
                onClick={() => setStep(step - 1)}
              >
                Back
              </button>
              {step < STEPS.length - 1 ? (
                <button
                  type="button"
                  className={experiment ? "secondary-button" : undefined}
                  onClick={() => setStep(step + 1)}
                >
                  Next step
                </button>
              ) : null}
              <button
                type="button"
                className={
                  (experiment ? dirty : step === STEPS.length - 1)
                    ? undefined
                    : "secondary-button"
                }
                disabled={
                  !projectId || save.isPending || (!!experiment && !dirty)
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
            {actions}
          </div>
          <p className="eval-section-description">
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
  return (
    <EvalFrame
      title="New experiment"
      description="Compare two versions on a shared dataset. Save, prepare, then start when ready."
      action={<Link to="/evals/datasets">Manage datasets</Link>}
    >
      <EvalSetupForm />
    </EvalFrame>
  );
}
