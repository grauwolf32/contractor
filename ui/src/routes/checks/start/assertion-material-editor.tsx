import { useEffect, useRef, useState } from "react";
import { schemaLimit, schemaViolation } from "../../../api/contract";
import { usePublicAPI } from "../../../api/context";
import { getProjectArtifactMetadata } from "../../../api/project-artifacts";
import { ErrorNotice } from "../../../app/error-notice";
import { Button, Field } from "../../../ui";
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
  const [open, setOpen] = useState(false);
  const [statements, setStatements] = useState("");
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
    },
    [],
  );
  const claims = statements
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
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
        items: claims.map((statement, index) => ({
          key: `assertion-${index + 1}`,
          version,
          statement,
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
    } catch (reason) {
      if (generation.current === current) setError(reason);
    } finally {
      if (generation.current === current) setBusy(false);
    }
  }
  return (
    <div className="ui-stack">
      <Button size="sm" disabled={busy} onClick={() => setOpen(!open)}>
        {open ? "Close checklist editor" : "Write a checklist"}
      </Button>
      {open ? (
        <>
          <Field
            label="Assertions"
            required
            hint="One observable claim per line, up to 128. Each keeps its place in the original checklist."
          >
            {(control) => (
              <textarea
                {...control}
                value={statements}
                disabled={busy}
                onChange={(event) => setStatements(event.target.value)}
              />
            )}
          </Field>
          <Field label="Checklist version" required>
            {(control) => (
              <input
                {...control}
                value={version}
                disabled={busy}
                onChange={(event) => setVersion(event.target.value)}
              />
            )}
          </Field>
          <Field label="Conditions" optional>
            {(control) => (
              <textarea
                {...control}
                value={conditions}
                disabled={busy}
                onChange={(event) => setConditions(event.target.value)}
              />
            )}
          </Field>
          <Field label="Why check these?" required>
            {(control) => (
              <textarea
                {...control}
                value={grounds}
                disabled={busy}
                onChange={(event) => setGrounds(event.target.value)}
              />
            )}
          </Field>
          <Field
            label="Evidence needed"
            required
            hint="One evidence requirement per line. Missing evidence remains a gap."
          >
            {(control) => (
              <textarea
                {...control}
                value={evidence}
                disabled={busy}
                onChange={(event) => setEvidence(event.target.value)}
              />
            )}
          </Field>
          <Button
            size="sm"
            disabled={
              busy ||
              claims.length === 0 ||
              claims.length > MAX_ASSERTIONS ||
              grounds.trim() === "" ||
              version.trim() === "" ||
              requirements.length === 0
            }
            onClick={() => void save()}
          >
            {busy ? "Saving checklist…" : "Save and attach checklist"}
          </Button>
          {error === undefined ? null : <ErrorNotice error={error} />}
        </>
      ) : null}
    </div>
  );
}
