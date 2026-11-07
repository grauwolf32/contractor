import { useEffect, useRef, useState } from "react";
import { schemaLimit, schemaViolation } from "../../../api/contract";
import { usePublicAPI } from "../../../api/context";
import { getProjectArtifactMetadata } from "../../../api/project-artifacts";
import { Icon } from "../../../app/icon";
import { ErrorNotice } from "../../../app/error-notice";
import { Button, Field, TechnicalDetails } from "../../../ui";
import { retainAuthoredMaterial } from "./authored-material";
import type { StartCheck } from "./use-start-check";

const MAX_ASSERTIONS = schemaLimit(
  "AuditAssertionsSource",
  "maxItems",
  "properties",
  "items",
);
const MAX_SOURCE_BYTES = schemaLimit(
  "AuditAssertionsSource",
  "x-max-document-bytes",
);

/** Author a finite checklist as an ordinary exact Project material. */
export function AssertionMaterialEditor({
  model,
  input,
}: {
  model: StartCheck;
  input: string;
}) {
  const api = usePublicAPI();
  const { setChecklistEditorOpen } = model;
  const [open, setOpen] = useState(false);
  const nextKey = useRef(2);
  const [entries, setEntries] = useState([
    { key: "assertion-1", statement: "" },
  ]);
  const [conditions, setConditions] = useState("");
  const [grounds, setGrounds] = useState("");
  const [version, setVersion] = useState("1");
  const [evidence, setEvidence] = useState(
    "Exact source evidence supporting the stated conclusion",
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>();
  const generation = useRef(0);
  useEffect(
    () => () => {
      generation.current++;
      setChecklistEditorOpen(false);
    },
    [setChecklistEditorOpen],
  );
  const validEntries = entries.every((entry) => entry.statement.trim() !== "");
  const requirements = evidence
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  async function save() {
    const current = generation.current;
    const ensureCurrent = () => {
      if (generation.current !== current)
        throw new Error("The selected check changed.");
    };
    setBusy(true);
    setError(undefined);
    try {
      const document = {
        schema: "contractor.audit.assertions.v1",
        items: entries.map(({ key, statement }) => ({
          key,
          version,
          statement: statement.trim(),
          grounds,
          scope: conditions.trim() === "" ? {} : { conditions },
          applicability: "within selected source and stated conditions",
          allowed_methods: ["static"],
          required_evidence: [...new Set(requirements)].sort(),
        })),
      };
      if (schemaViolation("AuditAssertionsSource", document) !== undefined)
        throw new Error(
          "Check the statements, version, conditions and evidence requirements.",
        );
      const payload = new Blob([JSON.stringify(document)], {
        type: "application/json",
      });
      if (payload.size > MAX_SOURCE_BYTES)
        throw new Error("Use a checklist no larger than 1 MiB.");
      const ref = await retainAuthoredMaterial(
        api,
        model.projectId,
        "materials",
        "assertions",
        payload,
        ensureCurrent,
      );
      const metadata = await getProjectArtifactMetadata(api, {
        projectId: model.projectId,
        ...ref,
      });
      ensureCurrent();
      model.registerAuthoredMaterial(input, metadata);
      setOpen(false);
      setChecklistEditorOpen(false);
    } catch (reason) {
      if (generation.current === current) setError(reason);
    } finally {
      if (generation.current === current) setBusy(false);
    }
  }
  return (
    <div className="start-checklist">
      <Button
        variant="ghost"
        size="sm"
        disabled={busy}
        onClick={() => {
          setOpen(!open);
          setChecklistEditorOpen(!open);
        }}
      >
        <Icon name="checklist" size={16} />
        {open ? "Close checklist editor" : "Write a checklist"}
      </Button>
      {open ? (
        <fieldset className="start-checklist-editor" disabled={busy}>
          <div className="start-checklist-heading">
            <div>
              <h4>Your checklist</h4>
              <p>One observable claim per check. Add only what you need.</p>
            </div>
            <span className="start-checklist-count">
              {entries.length} / {MAX_ASSERTIONS}
            </span>
          </div>
          <ol className="start-checklist-items">
            {entries.map((entry, index) => (
              <li key={entry.key}>
                <span className="start-checklist-number" aria-hidden="true">
                  {index + 1}
                </span>
                <Field label={`Check ${index + 1}`} required>
                  {(control) => (
                    <textarea
                      {...control}
                      rows={2}
                      value={entry.statement}
                      placeholder="For example: a user can read another project's export"
                      onChange={(event) =>
                        setEntries((current) =>
                          current.map((row) =>
                            row.key === entry.key
                              ? { ...row, statement: event.target.value }
                              : row,
                          ),
                        )
                      }
                    />
                  )}
                </Field>
                {entries.length > 1 ? (
                  <Button
                    variant="ghost"
                    size="xs"
                    aria-label={`Remove check ${index + 1}`}
                    onClick={() =>
                      setEntries((current) =>
                        current.filter((row) => row.key !== entry.key),
                      )
                    }
                  >
                    <Icon name="delete" size={15} />
                  </Button>
                ) : null}
              </li>
            ))}
          </ol>
          <Button
            variant="ghost"
            size="sm"
            disabled={entries.length >= MAX_ASSERTIONS}
            onClick={() => {
              const key = `assertion-${nextKey.current++}`;
              setEntries((current) => [...current, { key, statement: "" }]);
            }}
          >
            <Icon name="plus" size={15} />
            Add check
          </Button>
          <Field
            label="Why check these?"
            required
            hint="Explain the reason for this checklist. It is saved with every check."
          >
            {(control) => (
              <textarea
                {...control}
                rows={2}
                value={grounds}
                placeholder="For example: exports contain private project data"
                onChange={(event) => setGrounds(event.target.value)}
              />
            )}
          </Field>
          <TechnicalDetails summary="Conditions & evidence">
            <div className="start-checklist-options">
              <p className="ui-caption">
                Shared by these checks. Source checks use static analysis.
              </p>
              <Field label="Conditions" optional>
                {(control) => (
                  <textarea
                    {...control}
                    rows={2}
                    value={conditions}
                    onChange={(event) => setConditions(event.target.value)}
                  />
                )}
              </Field>
              <Field
                label="Evidence needed"
                required
                hint="One requirement per line. Missing evidence remains a gap."
              >
                {(control) => (
                  <textarea
                    {...control}
                    rows={2}
                    value={evidence}
                    onChange={(event) => setEvidence(event.target.value)}
                  />
                )}
              </Field>
              <Field label="Checklist version" required>
                {(control) => (
                  <input
                    {...control}
                    value={version}
                    onChange={(event) => setVersion(event.target.value)}
                  />
                )}
              </Field>
            </div>
          </TechnicalDetails>
          <div className="start-checklist-footer">
            <span className="ui-caption">
              Saves a project material. The check starts separately.
            </span>
            <Button
              size="sm"
              pending={busy}
              pendingLabel="Saving checklist…"
              disabled={
                !validEntries ||
                grounds.trim() === "" ||
                version.trim() === "" ||
                requirements.length === 0
              }
              onClick={() => void save()}
            >
              Save and attach checklist
            </Button>
          </div>
          {error === undefined ? null : <ErrorNotice error={error} />}
        </fieldset>
      ) : null}
    </div>
  );
}
