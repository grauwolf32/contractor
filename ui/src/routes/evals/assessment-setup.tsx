import type { EvalCapabilities, EvalDraft, EvalCheck } from "../../api/evals";
import { EvalField } from "./common";
import { MAX_EVAL_MEMBERS, MAX_EVAL_REPETITIONS } from "./setup-model";

const CHECK_LABELS: Record<string, string> = {
  "required-artifact@1": "Required output artifacts",
  "media-type@1": "Output media type",
  "json-schema@1": "Registered JSON schema",
  "human-review@1": "Human review",
};

export function AssessmentSetup({
  draft,
  onChange,
  capabilities,
}: {
  draft: EvalDraft;
  onChange: (draft: EvalDraft) => void;
  capabilities: EvalCapabilities | undefined;
}) {
  function updateCheck(index: number, patch: Partial<EvalCheck>) {
    onChange({
      ...draft,
      checks: draft.checks.map((check, i) =>
        i === index ? { ...check, ...patch } : check,
      ),
    });
  }
  function budget<K extends keyof EvalDraft["budgets"]>(
    key: K,
    value: EvalDraft["budgets"][K],
  ) {
    onChange({ ...draft, budgets: { ...draft.budgets, [key]: value } });
  }
  return (
    <>
      <h2>Assessment and repetitions</h2>
      <p>
        Checks assess their declared property. Human review uses a private
        rubric saved with the selected dataset revision.
      </p>
      {draft.checks.map((check, index) => (
        <fieldset className="eval-check" key={index}>
          <legend>Check {index + 1}</legend>
          <div className="form-grid">
            <EvalField label="Check ID">
              <input
                value={check.id}
                onChange={(e) => updateCheck(index, { id: e.target.value })}
              />
            </EvalField>
            <EvalField label="Evaluator">
              <select
                value={check.evaluator}
                onChange={(e) => {
                  const entry = capabilities?.checks.find(
                    (c) => c.evaluator === e.target.value,
                  );
                  const next: EvalCheck = {
                    id: check.id,
                    evaluator: e.target.value,
                    required: check.required,
                    ...(entry
                      ? { implementationSha256: entry.implementationSha256 }
                      : {}),
                  };
                  onChange({
                    ...draft,
                    checks: draft.checks.map((c, i) =>
                      i === index ? next : c,
                    ),
                  });
                }}
              >
                <option value="">Choose an evaluator</option>
                {capabilities?.checks.map((c) => (
                  <option
                    value={c.evaluator}
                    key={c.evaluator}
                    disabled={!c.available}
                  >
                    {CHECK_LABELS[c.evaluator] ?? c.evaluator}
                    {c.reason ? ` · ${c.reason}` : ""}
                  </option>
                ))}
              </select>
            </EvalField>
          </div>
          <label>
            <input
              type="checkbox"
              checked={check.required}
              onChange={(e) =>
                updateCheck(index, { required: e.target.checked })
              }
            />
            Required for quality to pass
          </label>
          {check.evaluator === "human-review@1" ? (
            <EvalField
              label="Pinned rubric revision"
              hint="Check ID and revision must match the private rubric in this dataset."
            >
              <input
                value={check.rubricRevision ?? ""}
                onChange={(e) =>
                  updateCheck(index, { rubricRevision: e.target.value })
                }
              />
            </EvalField>
          ) : (
            <>
              <EvalField
                label="Checked output role"
                hint={
                  check.evaluator === "required-artifact@1"
                    ? "Leave empty to check all required case outputs."
                    : undefined
                }
              >
                <input
                  value={check.parameters?.output ?? ""}
                  onChange={(e) =>
                    updateCheck(index, {
                      parameters: {
                        ...check.parameters,
                        output: e.target.value,
                      },
                    })
                  }
                />
              </EvalField>
              {check.evaluator === "media-type@1" ? (
                <EvalField label="Required media type">
                  <input
                    value={check.parameters?.mediaType ?? ""}
                    onChange={(e) =>
                      updateCheck(index, {
                        parameters: {
                          ...check.parameters,
                          mediaType: e.target.value,
                        },
                      })
                    }
                  />
                </EvalField>
              ) : null}
              {check.evaluator === "json-schema@1" ? (
                <EvalField label="Registered schema">
                  <select
                    value={check.parameters?.schema ?? ""}
                    onChange={(e) =>
                      updateCheck(index, {
                        parameters: {
                          ...check.parameters,
                          schema: e.target.value,
                        },
                      })
                    }
                  >
                    <option value="">Choose a schema</option>
                    {capabilities?.schemas?.map((schema) => (
                      <option key={schema}>{schema}</option>
                    ))}
                  </select>
                </EvalField>
              ) : null}
            </>
          )}
          <button
            type="button"
            className="secondary-button"
            onClick={() =>
              onChange({
                ...draft,
                checks: draft.checks.filter((_, i) => i !== index),
              })
            }
          >
            Remove check
          </button>
        </fieldset>
      ))}
      <button
        type="button"
        className="secondary-button"
        onClick={() => {
          const evaluator =
            capabilities?.checks.find(
              (c) => c.available && c.evaluator === "required-artifact@1",
            ) ?? capabilities?.checks.find((c) => c.available);
          onChange({
            ...draft,
            checks: [
              ...draft.checks,
              {
                id: `check-${draft.checks.length + 1}`,
                evaluator: evaluator?.evaluator ?? "",
                required: true,
                ...(evaluator
                  ? { implementationSha256: evaluator.implementationSha256 }
                  : {}),
              },
            ],
          });
        }}
      >
        Add assessment check
      </button>
      <div className="form-grid">
        <EvalField label="Repetitions">
          <input
            type="number"
            min={1}
            max={MAX_EVAL_REPETITIONS}
            value={draft.repetitions}
            onChange={(e) =>
              onChange({ ...draft, repetitions: Number(e.target.value) })
            }
          />
        </EvalField>
        <EvalField label="Concurrent members">
          <input
            type="number"
            min={1}
            max={draft.budgets.maxMembers}
            value={draft.budgets.maxInFlight}
            onChange={(e) => budget("maxInFlight", Number(e.target.value))}
          />
        </EvalField>
        <EvalField label="Maximum members">
          <input
            type="number"
            min={1}
            max={MAX_EVAL_MEMBERS}
            value={draft.budgets.maxMembers}
            onChange={(e) => budget("maxMembers", Number(e.target.value))}
          />
        </EvalField>
        <EvalField
          label="Time allowance (minutes)"
          hint="Pause and resume retain this original allowance."
        >
          <input
            type="number"
            min={1}
            value={draft.budgets.wallMs / 60000}
            onChange={(e) => budget("wallMs", Number(e.target.value) * 60000)}
          />
        </EvalField>
        <EvalField
          label="Observed token threshold (optional)"
          hint="Stops new dispatch after observed usage reaches the threshold; active work may overshoot. This is not a hard spend cap."
        >
          <input
            type="number"
            min={0}
            value={draft.budgets.maxObservedTotalTokens ?? ""}
            onChange={(e) =>
              budget(
                "maxObservedTotalTokens",
                e.target.value === "" ? null : Number(e.target.value),
              )
            }
          />
        </EvalField>
        <EvalField label="Execution order">
          <select
            value={draft.order.kind}
            onChange={(e) =>
              onChange({
                ...draft,
                order:
                  e.target.value === "alternating"
                    ? { kind: "alternating" }
                    : { kind: "seeded_shuffle", seed: 0 },
              })
            }
          >
            <option value="alternating">Alternate A / B</option>
            <option value="seeded_shuffle">Shuffle with explicit seed</option>
          </select>
        </EvalField>
        {draft.order.kind === "seeded_shuffle" ? (
          <EvalField label="Order seed">
            <input
              type="number"
              value={draft.order.seed ?? 0}
              onChange={(e) =>
                onChange({
                  ...draft,
                  order: {
                    kind: "seeded_shuffle",
                    seed: Number(e.target.value),
                  },
                })
              }
            />
          </EvalField>
        ) : null}
      </div>
      <h3>Comparison gates</h3>
      <div className="form-grid">
        {(
          [
            {
              key: "minCandidateEndToEndPass",
              label: "Minimum candidate pass fraction",
            },
            { key: "maxQualityDrop", label: "Maximum quality drop" },
          ] as const
        ).map(({ key, label }) => (
          <EvalField label={label} key={key}>
            <input
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={draft.comparison.gates[key]}
              onChange={(e) =>
                onChange({
                  ...draft,
                  comparison: {
                    ...draft.comparison,
                    gates: {
                      ...draft.comparison.gates,
                      [key]: Number(e.target.value),
                    },
                  },
                })
              }
            />
          </EvalField>
        ))}
        <EvalField label="Maximum token ratio B / A (optional)">
          <input
            type="number"
            min={0}
            step={0.01}
            value={draft.comparison.gates.maxTotalTokensRatio ?? ""}
            onChange={(e) => {
              const gates = { ...draft.comparison.gates };
              if (e.target.value === "") delete gates.maxTotalTokensRatio;
              else gates.maxTotalTokensRatio = Number(e.target.value);
              onChange({
                ...draft,
                comparison: { ...draft.comparison, gates },
              });
            }}
          />
        </EvalField>
      </div>
    </>
  );
}
