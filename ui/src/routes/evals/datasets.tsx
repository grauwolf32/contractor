import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useSearchParams } from "react-router";
import { usePublicAPI } from "../../api/context";
import {
  importEvalDataset,
  type EvalDataset,
  type EvalDatasetInput,
  type EvalCase,
} from "../../api/evals";
import { EvalFrame, EvalField, EvalError, KeyValueEditor } from "./common";
import { EvalArtifactPicker } from "./artifact-picker";
import { useEvalDatasets, useEvalOwner, useEvalProjects } from "./queries";
import { finishMutation, mutationKey } from "./recovery";
import { MAX_EVAL_CASES, MAX_EVAL_DOCUMENT_BYTES } from "./setup-model";

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
  const [inputRole, setInputRole] = useState("");
  const [pick, setPick] = useState(false);
  const [outputRole, setOutputRole] = useState("");
  const [mediaType, setMediaType] = useState("application/json");
  return (
    <div className="eval-case-editor">
      <div className="form-grid">
        <EvalField label="Case ID">
          <input
            value={value.id}
            onChange={(e) => onChange({ ...value, id: e.target.value })}
            required
            maxLength={128}
          />
        </EvalField>
        <EvalField label="Task kind">
          <input
            value={value.task.kind}
            onChange={(e) =>
              onChange({
                ...value,
                task: { ...value.task, kind: e.target.value },
              })
            }
            required
          />
        </EvalField>
      </div>
      <EvalField
        label="Visible task objective"
        hint="This task and its inputs are sent to the selected Workflow or Audit."
      >
        <textarea
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
      </EvalField>
      <KeyValueEditor
        label="Task parameters"
        value={value.task.parameters}
        onChange={(parameters) =>
          onChange({ ...value, task: { ...value.task, parameters } })
        }
      />
      <EvalField
        label="Required capabilities"
        hint="Optional exact capability names, separated by commas."
      >
        <input
          value={value.requires.join(", ")}
          onChange={(e) =>
            onChange({
              ...value,
              requires: e.target.value
                .split(",")
                .map((x) => x.trim())
                .filter(Boolean),
            })
          }
        />
      </EvalField>
      <h4>Exact input artifacts</h4>
      {Object.entries(value.inputs).map(([role, ref]) => (
        <p key={role}>
          {role}: {ref.namespace}/{ref.name} · {ref.revision}{" "}
          <button
            type="button"
            className="secondary"
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
          </button>
        </p>
      ))}
      <div className="eval-actions">
        <EvalField label="Input role">
          <input
            value={inputRole}
            onChange={(e) => setInputRole(e.target.value)}
          />
        </EvalField>
        <button
          type="button"
          className="secondary"
          disabled={!inputRole.trim()}
          onClick={() => setPick(!pick)}
        >
          Choose exact input
        </button>
      </div>
      {pick ? (
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
      ) : null}
      <h4>Expected output roles</h4>
      {Object.entries(value.outputs).map(([role, output]) => (
        <p key={role}>
          {role} · {output.mediaTypes.join(", ")}{" "}
          <label>
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
            Required output
          </label>
          <button
            type="button"
            className="secondary"
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
          </button>
        </p>
      ))}
      <div className="form-grid">
        <EvalField label="Output role">
          <input
            value={outputRole}
            onChange={(e) => setOutputRole(e.target.value)}
          />
        </EvalField>
        <EvalField label="Output media types">
          <input
            value={mediaType}
            onChange={(e) => setMediaType(e.target.value)}
          />
        </EvalField>
      </div>
      <button
        type="button"
        className="secondary"
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
      </button>
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
  const [data, setData] = useState<EvalDatasetInput>({
    datasetId: "",
    name: "",
    cases: [newCase()],
    privateChecks: [],
  });
  const [error, setError] = useState<Error | null>(null);
  const [includePrivate, setIncludePrivate] = useState(false);
  const [imported, setImported] = useState<EvalDatasetInput | null>(null);
  const save = useMutation({
    mutationFn: async () => {
      const body = imported
        ? {
            ...imported,
            privateChecks: includePrivate ? (imported.privateChecks ?? []) : [],
          }
        : data;
      const operation = `dataset:${projectId}`;
      const result = await importEvalDataset(
        api,
        projectId,
        body,
        await mutationKey(owner, operation, body),
      );
      await finishMutation(owner, operation, body);
      return result;
    },
    onSuccess: async (result) => {
      await cache.invalidateQueries({
        queryKey: ["evals", "datasets", projectId],
      });
      onSaved(result);
    },
  });
  async function importFile(file?: File) {
    setError(null);
    if (!file) return;
    try {
      if (file.size > MAX_EVAL_DOCUMENT_BYTES)
        throw new Error("Dataset file exceeds 16 MiB.");
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
      setImported(candidate);
      setIncludePrivate(false);
    } catch (cause) {
      setError(
        cause instanceof Error ? cause : new Error("Cannot read the dataset."),
      );
    }
  }
  return (
    <section className="panel eval-panel">
      <h2>Create dataset revision</h2>
      <p>
        Revisions are immutable. Import retains source provenance; changing
        visible cases requires a new revision.
      </p>
      <EvalField label="Import dataset JSON">
        <input
          type="file"
          accept="application/json,.json"
          onChange={(e) => void importFile(e.target.files?.[0])}
        />
      </EvalField>
      {imported ? (
        <>
          <p>
            {imported.name} · {imported.cases.length} cases
          </p>
          <label>
            <input
              type="checkbox"
              checked={includePrivate}
              onChange={(e) => setIncludePrivate(e.target.checked)}
            />
            Include private assessment rubrics and expectations in the evaluator
            partition
          </label>
          <p>
            Private checks are excluded from imports by default and never become
            model inputs.
          </p>
          <button
            type="button"
            className="secondary"
            onClick={() => setImported(null)}
          >
            Author cases instead
          </button>
        </>
      ) : (
        <>
          <div className="form-grid">
            <EvalField label="Dataset ID">
              <input
                value={data.datasetId}
                onChange={(e) =>
                  setData({ ...data, datasetId: e.target.value })
                }
              />
            </EvalField>
            <EvalField label="Dataset name">
              <input
                value={data.name}
                onChange={(e) => setData({ ...data, name: e.target.value })}
              />
            </EvalField>
          </div>
          {data.cases.map((value, index) => (
            <details
              className="eval-case"
              key={index}
              open={data.cases.length === 1 || undefined}
            >
              <summary>
                Case {index + 1}: {value.id || "New case"}
              </summary>
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
              {data.cases.length > 1 ? (
                <button
                  type="button"
                  className="secondary"
                  onClick={() =>
                    setData({
                      ...data,
                      cases: data.cases.filter((_, i) => i !== index),
                    })
                  }
                >
                  Remove case {index + 1}
                </button>
              ) : null}
            </details>
          ))}
          <button
            type="button"
            className="secondary"
            disabled={data.cases.length >= MAX_EVAL_CASES}
            onClick={() =>
              setData({ ...data, cases: [...data.cases, newCase()] })
            }
          >
            Add case
          </button>
          <details>
            <summary>Private human review rubrics</summary>
            <p>
              Only the evaluator and owner review screen can read these rubrics.
              They are separate from visible cases.
            </p>
            {data.privateChecks.map((check, index) => (
              <fieldset key={index}>
                <legend>Rubric {index + 1}</legend>
                <div className="form-grid">
                  <EvalField label="Review check ID">
                    <input
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
                  </EvalField>
                  <EvalField label="Rubric revision">
                    <input
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
                  </EvalField>
                </div>
                <EvalField label="Private rubric">
                  <textarea
                    rows={4}
                    value={check.rubric}
                    onChange={(e) =>
                      setData({
                        ...data,
                        privateChecks: data.privateChecks.map((c, i) =>
                          i === index ? { ...c, rubric: e.target.value } : c,
                        ),
                      })
                    }
                  />
                </EvalField>
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
                <button
                  type="button"
                  className="secondary"
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
                </button>
              </fieldset>
            ))}
            <button
              type="button"
              className="secondary"
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
            </button>
          </details>
        </>
      )}
      <EvalError error={error ?? save.error} />
      <button
        type="button"
        disabled={save.isPending}
        onClick={() => save.mutate()}
      >
        {save.isPending ? "Saving revision…" : "Save dataset revision"}
      </button>
    </section>
  );
}

export function EvalDatasetsRoute() {
  const [params, setParams] = useSearchParams();
  const projects = useEvalProjects();
  const projectId = params.get("project") ?? "";
  const datasets = useEvalDatasets(projectId);
  const [author, setAuthor] = useState(false);
  return (
    <EvalFrame
      title="Evaluation datasets"
      action={
        <Link className="button-link" to="/evals/new">
          New experiment
        </Link>
      }
    >
      <EvalField label="Evaluation workspace">
        <select
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
      </EvalField>
      <EvalError error={projects.error ?? datasets.error} />
      {projectId ? (
        <>
          <ul className="eval-choice-list">
            {datasets.data?.map((d) => (
              <li key={d.revision}>
                <span>
                  <strong>{d.name}</strong>
                  <small>
                    {d.caseCount} cases · {d.revision}
                    {d.source
                      ? ` · ${d.source.system}/${d.source.id}`
                      : " · Native"}
                  </small>
                </span>
                <Link
                  to={`/evals/new?project=${encodeURIComponent(projectId)}&dataset=${encodeURIComponent(d.datasetId)}&revision=${encodeURIComponent(d.revision)}`}
                >
                  Use revision
                </Link>
              </li>
            ))}
          </ul>
          <button type="button" onClick={() => setAuthor(!author)}>
            Create or import dataset
          </button>
          {author ? (
            <DatasetAuthor
              projectId={projectId}
              onSaved={() => setAuthor(false)}
            />
          ) : null}
        </>
      ) : (
        <p>Select an evaluation workspace to manage its datasets.</p>
      )}
    </EvalFrame>
  );
}
