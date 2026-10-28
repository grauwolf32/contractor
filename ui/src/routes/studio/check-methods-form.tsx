import { Button, Field } from "../../ui";
import { DraftField, DraftList } from "./advanced-fields";
import { record, textValue, uniqueName, type Path } from "./document";
import type { FormEditor } from "./form-editor";
import {
  CHECK_METHODS,
  MAX_CHECK_METHODS,
  MAX_CHECK_CAPABILITIES,
  FINDING_CHECK_PLAN_SCHEMA,
} from "./check-methods";

export function CheckMethodsForm({ editor }: { editor: FormEditor }) {
  const spec = record(editor.draft.value.spec),
    methods = record(spec.methods);
  const roles = Object.entries(record(spec.workflows))
    .filter(([, role]) => record(role).kind === "check")
    .map(([name]) => name);
  const available = CHECK_METHODS.filter(
    (method) =>
      !Object.values(methods).some((entry) => record(entry).method === method),
  );
  return (
    <>
      <h3>Proposed check methods</h3>
      <p>Choose which check workflow handles each registered method.</p>
      {Object.entries(methods).map(([name, raw]) => {
        const value = record(raw),
          path: Path = ["spec", "methods", name];
        const capabilities = Array.isArray(value.requiredCapabilities)
          ? value.requiredCapabilities
          : [];
        return (
          <div className="studio-group" key={name}>
            <strong className="studio-identity">{name}</strong>
            <Field label="Method" className="studio-field">
              {(control) => (
                <select
                  {...control}
                  value={textValue(value.method)}
                  onChange={(event) =>
                    editor.onPatch([...path, "method"], event.target.value)
                  }
                >
                  {CHECK_METHODS.map((method) => (
                    <option key={method} value={method}>
                      {method}
                    </option>
                  ))}
                </select>
              )}
            </Field>
            <Field label="Check workflow role" className="studio-field">
              {(control) => (
                <select
                  {...control}
                  value={textValue(value.workflowRole)}
                  onChange={(event) =>
                    editor.onPatch(
                      [...path, "workflowRole"],
                      event.target.value,
                    )
                  }
                >
                  {roles.map((role) => (
                    <option key={role}>{role}</option>
                  ))}
                </select>
              )}
            </Field>
            {capabilities.map((_, index) => {
              const p: Path = [...path, "requiredCapabilities", index];
              return (
                <div className="studio-group" key={index}>
                  <DraftField
                    editor={editor}
                    label="Stage"
                    path={[...p, "stage"]}
                  />
                  <DraftField
                    editor={editor}
                    label="Worker"
                    path={[...p, "agent"]}
                  />
                  <DraftField
                    editor={editor}
                    label="Toolset"
                    path={[...p, "ref", "toolsetId"]}
                  />
                  <DraftField
                    editor={editor}
                    label="Toolset version"
                    path={[...p, "ref", "version"]}
                  />
                  <DraftList
                    editor={editor}
                    label="Required tools"
                    path={[...p, "tools"]}
                  />
                  <Button
                    onClick={() =>
                      editor.onPatch(
                        [...path, "requiredCapabilities"],
                        capabilities.filter(
                          (_, position) => position !== index,
                        ),
                      )
                    }
                  >
                    Remove required tools
                  </Button>
                </div>
              );
            })}
            <Button
              disabled={capabilities.length >= MAX_CHECK_CAPABILITIES}
              onClick={() =>
                editor.onPatch(
                  [...path, "requiredCapabilities"],
                  [
                    ...capabilities,
                    {
                      stage: "check",
                      agent: "worker",
                      ref: { toolsetId: "run-artifacts", version: "1" },
                      tools: ["read_artifact"],
                    },
                  ],
                )
              }
            >
              Add required tools
            </Button>
            <details>
              <summary>Technical details</summary>
              <p className="studio-identity">{textValue(value.planSchema)}</p>
            </details>
            <Button onClick={() => editor.onRemove(path, `method ${name}`)}>
              Remove method
            </Button>
          </div>
        );
      })}
      <Button
        disabled={
          !available.length ||
          !roles.length ||
          Object.keys(methods).length >= MAX_CHECK_METHODS
        }
        onClick={() => {
          const method = available[0]!,
            name = uniqueName(methods, "method");
          editor.onPatch(["spec", "methods", name], {
            planSchema: FINDING_CHECK_PLAN_SCHEMA,
            method,
            workflowRole: roles[0],
            requiredCapabilities: [],
          });
        }}
      >
        Add method route
      </Button>
    </>
  );
}
