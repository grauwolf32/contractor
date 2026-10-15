import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useId, useState } from "react";
import { useSearchParams } from "react-router";
import { usePublicAPI } from "../../api/context";
import {
  importEvalDataset,
  type EvalDataset,
  type EvalDatasetInput,
  type EvalCase,
} from "../../api/evals";
import { queryKeys } from "../../api/query-keys";
import { useDocumentTitle } from "../../app/document-title";
import { QueryView } from "../../app/query-view";
import { Button, ButtonLink, EmptyState, Field, IdChip } from "../../ui";
import { EvalArtifactPicker } from "./artifact-picker";
import {
  CommaSeparatedInput,
  EvalFrame,
  EvalError,
  KeyValueEditor,
} from "./common";
import {
  KeyValueValidityProvider,
  useKeyValueValidity,
} from "./key-value-validity";
import { useEvalDatasets, useEvalOwner, useEvalProjects } from "./queries";
import { recoverableMutation } from "./recovery";
import { MAX_EVAL_DOCUMENT_BYTES } from "./plan-size";
import { MAX_EVAL_CASES } from "./setup-model";

const MAX_EVAL_IMPORT_FILE_BYTES = 16 * 1024 * 1024;
const DATASET_SIZE_ERROR =
  "Dataset exceeds the 1 MiB evaluation document limit; split it into several datasets.";

function datasetRequestBody(
  data: EvalDatasetInput,
  imported: EvalDatasetInput | null,
  includePrivate: boolean,
): EvalDatasetInput {
  return imported
    ? {
        ...imported,
        privateChecks: includePrivate ? (imported.privateChecks ?? []) : [],
      }
    : data;
}

function datasetRequestIsTooLarge(body: EvalDatasetInput): boolean {
  return (
    new TextEncoder().encode(JSON.stringify(body)).length >
    MAX_EVAL_DOCUMENT_BYTES
  );
}

function newCase(): EvalCase {
  return {
    id: "",
    task: { kind: "analysis", objective: "", parameters: {} },
    inputs: {},
    requires: [],
    outputs: {},
  };
}

function CaseEditor({
  value,
  onChange,
  projectId,
}: {
  value: EvalCase;
  onChange: (value: EvalCase) => void;
  projectId: string;
}) {
  const picker = useId();
  const [inputRole, setInputRole] = useState("");
  const [pick, setPick] = useState(false);
  const [outputRole, setOutputRole] = useState("");
  const [mediaType, setMediaType] = useState("application/json");
  return (
    <div className="eval-case-editor">
      <div className="eval-grid">
        <Field label="Case ID">
          {(control) => (
            <input
              {...control}
              value={value.id}
              onChange={(e) => onChange({ ...value, id: e.target.value })}
              required
              maxLength={128}
            />
          )}
        </Field>
        <Field label="Task kind">
          {(control) => (
            <input
              {...control}
              value={value.task.kind}
              onChange={(e) =>
                onChange({
                  ...value,
                  task: { ...value.task, kind: e.target.value },
                })
              }
              required
            />
          )}
        </Field>
      </div>
      <Field
        label="Visible task objective"
        hint="This task and its inputs are sent to the selected Workflow or check."
      >
        {(control) => (
          <textarea
            {...control}
            value={value.task.objective}
            onChange={(e) =>
              onChange({
                ...value,
                task: { ...value.task, objective: e.target.value },
              })
            }
            required
            rows={3}
          />
        )}
      </Field>
      <KeyValueEditor
        label="Task parameters"
        value={value.task.parameters}
        onChange={(parameters) =>
          onChange({ ...value, task: { ...value.task, parameters } })
        }
      />
      <Field
        label="Required capabilities"
        hint="Optional capability names, separated by commas."
      >
        {(control) => (
          <CommaSeparatedInput
            {...control}
            value={value.requires}
            onChange={(requires) =>
              onChange({
                ...value,
                requires,
              })
            }
          />
        )}
      </Field>
      <h4 className="eval-subheading">Input files</h4>
      {Object.keys(value.inputs).length ? (
        <ul className="eval-roles" role="list">
          {Object.entries(value.inputs).map(([role, ref]) => (
            <li key={role}>
              <span className="eval-role-main">
                <code>{role}</code> · {ref.namespace}/{ref.name}{" "}
                <IdChip value={ref.revision} label={`${role} revision`} />
              </span>
              <Button
                variant="ghost"
                size="sm"
                onClick={() =>
                  onChange({
                    ...value,
                    inputs: Object.fromEntries(
                      Object.entries(value.inputs).filter(
                        ([name]) => name !== role,
                      ),
                    ),
                  })
                }
              >
                Remove input {role}
              </Button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="ui-caption">No input files yet.</p>
      )}
      <div className="eval-inline-fields">
        <Field label="Input role">
          {(control) => (
            <input
              {...control}
              value={inputRole}
              onChange={(e) => setInputRole(e.target.value)}
            />
          )}
        </Field>
        <Button
          aria-expanded={pick}
          aria-controls={pick ? picker : undefined}
          disabled={!inputRole.trim()}
          onClick={() => setPick(!pick)}
        >
          Choose input
        </Button>
      </div>
      {pick ? (
        <div id={picker}>
          <EvalArtifactPicker
            projectId={projectId}
            onSelect={(ref) => {
              onChange({
                ...value,
                inputs: { ...value.inputs, [inputRole.trim()]: ref },
              });
              setPick(false);
              setInputRole("");
            }}
          />
        </div>
      ) : null}
      <h4 className="eval-subheading">Expected output roles</h4>
      {Object.keys(value.outputs).length ? (
        <ul className="eval-roles" role="list">
          {Object.entries(value.outputs).map(([role, output]) => (
            <li key={role}>
              <span className="eval-role-main">
                <code>{role}</code> · {output.mediaTypes.join(", ")}
              </span>
              <label className="eval-checkbox">
                <input
                  type="checkbox"
                  checked={output.required}
                  onChange={(e) =>
                    onChange({
                      ...value,
                      outputs: {
                        ...value.outputs,
                        [role]: { ...output, required: e.target.checked },
                      },
                    })
                  }
                />
                <span>Required output</span>
              </label>
              <Button
                variant="ghost"
                size="sm"
                onClick={() =>
                  onChange({
                    ...value,
                    outputs: Object.fromEntries(
                      Object.entries(value.outputs).filter(
                        ([name]) => name !== role,
                      ),
                    ),
                  })
                }
              >
                Remove output {role}
              </Button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="ui-caption">No output roles yet.</p>
      )}
      <div className="eval-grid">
        <Field label="Output role">
          {(control) => (
            <input
              {...control}
              value={outputRole}
              onChange={(e) => setOutputRole(e.target.value)}
            />
          )}
        </Field>
        <Field label="Output media types">
          {(control) => (
            <input
              {...control}
              value={mediaType}
              onChange={(e) => setMediaType(e.target.value)}
            />
          )}
        </Field>
      </div>
      <div className="ui-toolbar">
        <Button
          size="sm"
          disabled={!outputRole.trim() || !mediaType.trim()}
          onClick={() => {
            onChange({
              ...value,
              outputs: {
                ...value.outputs,
                [outputRole.trim()]: {
                  mediaTypes: mediaType.split(",").map((x) => x.trim()),
                  required: true,
                },
              },
            });
            setOutputRole("");
          }}
        >
          Add output role
        </Button>
      </div>
    </div>
  );
}

export function DatasetAuthor({
  projectId,
  onSaved,
}: {
  projectId: string;
  onSaved: (dataset: EvalDataset) => void;
}) {
  const api = usePublicAPI(),
    owner = useEvalOwner(),
    cache = useQueryClient();
  const heading = useId();
  const [data, setData] = useState<EvalDatasetInput>({
    datasetId: "",
    name: "",
    cases: [newCase()],
    privateChecks: [],
  });
  const [error, setError] = useState<Error | null>(null);
  const [includePrivate, setIncludePrivate] = useState(false);
  const [imported, setImported] = useState<EvalDatasetInput | null>(null);
  const [importRejected, setImportRejected] = useState(false);
  const keyValues = useKeyValueValidity();
  const body = datasetRequestBody(data, imported, includePrivate);
  const sizeProblem = datasetRequestIsTooLarge(body)
    ? DATASET_SIZE_ERROR
    : null;
  const save = useMutation({
    mutationFn: async () => {
      if (!imported && keyValues.invalid)
        throw new Error(
          "Resolve duplicate or empty parameter names before saving.",
        );
      if (datasetRequestIsTooLarge(body)) throw new Error(DATASET_SIZE_ERROR);
      const operation = `dataset:${projectId}`;
      return recoverableMutation(owner, operation, body, (key) =>
        importEvalDataset(api, projectId, body, key),
      );
    },
    onSuccess: async (result) => {
      await cache.invalidateQueries({
        queryKey: queryKeys.evals.datasets(projectId),
      });
      onSaved(result);
    },
  });
  async function importFile(file?: File) {
    if (!file) return;
    setError(null);
    setImported(null);
    setImportRejected(false);
    try {
      if (file.size > MAX_EVAL_IMPORT_FILE_BYTES)
        throw new Error("Dataset source file is too large to inspect.");
      const candidate = JSON.parse(await file.text()) as EvalDatasetInput;
      if (
        !candidate ||
        !Array.isArray(candidate.cases) ||
        !candidate.cases.length ||
        candidate.cases.length > MAX_EVAL_CASES ||
        typeof candidate.datasetId !== "string" ||
        typeof candidate.name !== "string"
      )
        throw new Error(
          "Choose a managed dataset document with a name, ID and visible cases.",
        );
      if (datasetRequestIsTooLarge(datasetRequestBody(data, candidate, false)))
        throw new Error(DATASET_SIZE_ERROR);
      setImported(candidate);
      setIncludePrivate(false);
    } catch (cause) {
      setImportRejected(true);
      setError(
        cause instanceof Error ? cause : new Error("Cannot read the dataset."),
      );
    }
  }
  return (
    <section className="eval-panel eval-author" aria-labelledby={heading}>
      <div className="eval-step-heading">
        <h2 id={heading}>Create dataset revision</h2>
        <p className="ui-caption">
          Changing visible cases requires a new revision.
        </p>
      </div>
      <Field label="Import dataset JSON">
        {(control) => (
          <input
            {...control}
            type="file"
            accept="application/json,.json"
            onChange={(e) => void importFile(e.target.files?.[0])}
          />
        )}
      </Field>
      {importRejected ? (
        <div className="ui-toolbar">
          <Button
            size="sm"
            onClick={() => {
              setImportRejected(false);
              setError(null);
            }}
          >
            Author cases instead
          </Button>
        </div>
      ) : imported ? (
        <div className="eval-import">
          {!sizeProblem ? (
            <p className="eval-import-summary">
              {imported.name} · {imported.cases.length} cases
            </p>
          ) : null}
          <label className="eval-checkbox">
            <input
              type="checkbox"
              checked={includePrivate}
              onChange={(e) => setIncludePrivate(e.target.checked)}
            />
            <span>
              Include private assessment rubrics and expectations in the
              evaluator partition
            </span>
          </label>
          <p className="ui-caption">
            Private rubrics are excluded from imports by default and never
            become model inputs.
          </p>
          <div className="ui-toolbar">
            <Button
              size="sm"
              onClick={() => {
                setImported(null);
                setError(null);
              }}
            >
              Author cases instead
            </Button>
          </div>
        </div>
      ) : (
        <>
          <div className="eval-grid">
            <Field label="Dataset ID">
              {(control) => (
                <input
                  {...control}
                  value={data.datasetId}
                  onChange={(e) =>
                    setData({ ...data, datasetId: e.target.value })
                  }
                />
              )}
            </Field>
            <Field label="Dataset name">
              {(control) => (
                <input
                  {...control}
                  value={data.name}
                  onChange={(e) => setData({ ...data, name: e.target.value })}
                />
              )}
            </Field>
          </div>
          <div className="ui-stack" data-gap="2">
            {data.cases.map((value, index) => (
              <details
                className="eval-case eval-disclosure"
                key={index}
                open={data.cases.length === 1 || undefined}
              >
                <summary>
                  Case {index + 1}: {value.id || "New case"}
                </summary>
                <div className="eval-disclosure-body">
                  <KeyValueValidityProvider value={keyValues.register}>
                    <CaseEditor
                      value={value}
                      projectId={projectId}
                      onChange={(next) =>
                        setData({
                          ...data,
                          cases: data.cases.map((item, i) =>
                            i === index ? next : item,
                          ),
                        })
                      }
                    />
                  </KeyValueValidityProvider>
                  {data.cases.length > 1 ? (
                    <div className="ui-toolbar">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() =>
                          setData({
                            ...data,
                            cases: data.cases.filter((_, i) => i !== index),
                          })
                        }
                      >
                        Remove case {index + 1}
                      </Button>
                    </div>
                  ) : null}
                </div>
              </details>
            ))}
          </div>
          <div className="ui-toolbar">
            <Button
              size="sm"
              disabled={data.cases.length >= MAX_EVAL_CASES}
              onClick={() =>
                setData({ ...data, cases: [...data.cases, newCase()] })
              }
            >
              Add case
            </Button>
          </div>
          <details className="eval-disclosure">
            <summary>Private human review rubrics</summary>
            <div className="eval-disclosure-body">
              <p className="ui-caption">
                Only the evaluator and owner review screen can read these
                rubrics. They are separate from visible cases.
              </p>
              {data.privateChecks.map((check, index) => (
                <fieldset className="eval-subfieldset" key={index}>
                  <legend>Rubric {index + 1}</legend>
                  <div className="eval-grid">
                    <Field label="Review criterion ID">
                      {(control) => (
                        <input
                          {...control}
                          value={check.id}
                          onChange={(e) =>
                            setData({
                              ...data,
                              privateChecks: data.privateChecks.map((c, i) =>
                                i === index ? { ...c, id: e.target.value } : c,
                              ),
                            })
                          }
                        />
                      )}
                    </Field>
                    <Field label="Rubric revision">
                      {(control) => (
                        <input
                          {...control}
                          value={check.revision}
                          onChange={(e) =>
                            setData({
                              ...data,
                              privateChecks: data.privateChecks.map((c, i) =>
                                i === index
                                  ? { ...c, revision: e.target.value }
                                  : c,
                              ),
                            })
                          }
                        />
                      )}
                    </Field>
                  </div>
                  <Field label="Private rubric">
                    {(control) => (
                      <textarea
                        {...control}
                        rows={4}
                        value={check.rubric}
                        onChange={(e) =>
                          setData({
                            ...data,
                            privateChecks: data.privateChecks.map((c, i) =>
                              i === index
                                ? { ...c, rubric: e.target.value }
                                : c,
                            ),
                          })
                        }
                      />
                    )}
                  </Field>
                  <KeyValueValidityProvider value={keyValues.register}>
                    <KeyValueEditor
                      label="Private expectations"
                      value={check.expected}
                      onChange={(expected) =>
                        setData({
                          ...data,
                          privateChecks: data.privateChecks.map((c, i) =>
                            i === index ? { ...c, expected } : c,
                          ),
                        })
                      }
                    />
                  </KeyValueValidityProvider>
                  <div className="ui-toolbar">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() =>
                        setData({
                          ...data,
                          privateChecks: data.privateChecks.filter(
                            (_, i) => i !== index,
                          ),
                        })
                      }
                    >
                      Remove rubric
                    </Button>
                  </div>
                </fieldset>
              ))}
              <div className="ui-toolbar">
                <Button
                  size="sm"
                  onClick={() =>
                    setData({
                      ...data,
                      privateChecks: [
                        ...data.privateChecks,
                        { id: "", revision: "", rubric: "", expected: {} },
                      ],
                    })
                  }
                >
                  Add human rubric
                </Button>
              </div>
            </div>
          </details>
        </>
      )}
      <EvalError
        error={error ?? (sizeProblem ? new Error(sizeProblem) : save.error)}
      />
      {!importRejected ? (
        <div className="ui-toolbar">
          <Button
            variant="primary"
            disabled={sizeProblem !== null || (!imported && keyValues.invalid)}
            pending={save.isPending}
            pendingLabel="Saving revision…"
            onClick={() => save.mutate()}
          >
            Save dataset revision
          </Button>
        </div>
      ) : null}
    </section>
  );
}

export function EvalDatasetsRoute() {
  useDocumentTitle("Evaluation datasets");
  const [params, setParams] = useSearchParams();
  const projects = useEvalProjects();
  const projectId = params.get("project") ?? "";
  const datasets = useEvalDatasets(projectId);
  const [author, setAuthor] = useState(false);
  const revisionsHeading = useId(),
    authorPanel = useId();
  return (
    <EvalFrame
      title="Evaluation datasets"
      breadcrumb={[
        { label: "Experiments", to: "/evals" },
        { label: "Datasets" },
      ]}
      description="Visible cases and private review rubrics, saved as immutable revisions of an evaluation workspace."
      action={
        <ButtonLink variant="primary" size="sm" to="/evals/new">
          New experiment
        </ButtonLink>
      }
    >
      <div className="eval-inline-fields">
        <Field label="Evaluation workspace">
          {(control) => (
            <select
              {...control}
              value={projectId}
              onChange={(e) => {
                setParams({ project: e.target.value });
                setAuthor(false);
              }}
            >
              <option value="">Choose a workspace</option>
              {projects.data
                ?.filter((p) => p.lifecycle === "active")
                .map((p) => (
                  <option key={p.projectId} value={p.projectId}>
                    {p.name}
                  </option>
                ))}
            </select>
          )}
        </Field>
      </div>
      <EvalError error={projects.error} />
      {projectId ? (
        <>
          <section className="eval-section" aria-labelledby={revisionsHeading}>
            <div className="eval-section-heading">
              <h2 id={revisionsHeading}>Dataset revisions</h2>
              {datasets.data ? (
                <span className="eval-count">{datasets.data.length}</span>
              ) : null}
            </div>
            <QueryView
              query={datasets}
              loading={<p role="status">Loading datasets…</p>}
              isEmpty={(data) => data.length === 0}
              empty={
                <EmptyState title="No datasets yet">
                  <p>
                    Author visible cases or import a dataset document to create
                    the first revision.
                  </p>
                </EmptyState>
              }
            >
              {(data) => (
                <ul className="eval-records" role="list">
                  {data.map((d) => (
                    <li className="eval-record eval-dataset" key={d.revision}>
                      <div className="eval-record-main">
                        <strong>{d.name}</strong>
                        <span className="eval-record-meta">
                          <span>
                            {d.caseCount} cases ·{" "}
                            {d.source
                              ? `${d.source.system}/${d.source.id}`
                              : "Native"}
                          </span>
                          <IdChip value={d.revision} label="dataset revision" />
                        </span>
                      </div>
                      <ButtonLink
                        size="sm"
                        to={`/evals/new?project=${encodeURIComponent(projectId)}&dataset=${encodeURIComponent(d.datasetId)}&revision=${encodeURIComponent(d.revision)}`}
                      >
                        Use revision
                      </ButtonLink>
                    </li>
                  ))}
                </ul>
              )}
            </QueryView>
          </section>
          <div className="ui-toolbar">
            <Button
              aria-expanded={author}
              aria-controls={author ? authorPanel : undefined}
              onClick={() => setAuthor(!author)}
            >
              Create or import dataset
            </Button>
          </div>
          {author ? (
            <div id={authorPanel}>
              <DatasetAuthor
                projectId={projectId}
                onSaved={() => setAuthor(false)}
              />
            </div>
          ) : null}
        </>
      ) : (
        <EmptyState title="Choose an evaluation workspace">
          <p>Select an evaluation workspace to manage its datasets.</p>
        </EmptyState>
      )}
    </EvalFrame>
  );
}
