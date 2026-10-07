import { appendDraft, at, record, uniqueName, type Path } from "./document";
import { DraftChoice, DraftField } from "./advanced-fields";
import { useDraftEdit, type FormEditor } from "./form-editor";

export const OVERLAY_MEDIA_TYPE =
  "application/vnd.contractor.workspace-overlay+json";
export function WorkspaceForm({
  editor,
  stagePath,
}: {
  editor: FormEditor;
  stagePath: Path;
}) {
  const { draft, onPatch, onRemove, onReplace } = editor;
  const path = [...stagePath, "context", "workspace"];
  const workspace = at(draft.value, path);
  const row = record(workspace);
  const inputs = record(
    at(draft.value, [...stagePath, "context", "artifacts"]),
  );
  const slots = Object.keys(inputs);
  const required = slots.filter(
    (name) => record(inputs[name]).required === true,
  );
  const results = record(
    at(draft.value, [...stagePath, "result", "artifacts"]),
  );
  const { edit, error } = useDraftEdit(draft, onReplace);
  const binding = (label: string, path: Path, options: string[]) => (
    <DraftChoice editor={editor} label={label} path={path} options={options} />
  );
  return (
    <details className="studio-settings" open>
      <summary>Workspace</summary>
      {workspace === undefined ? (
        <>
          <p>Mount stage input files into a workspace.</p>
          <button
            className="ui-btn"
            onClick={() =>
              onPatch(path, {
                mode: "direct",
                sources: [{ artifact: required[0] ?? "", target: "" }],
              })
            }
          >
            Configure workspace
          </button>
        </>
      ) : workspace === null ||
        typeof workspace !== "object" ||
        Array.isArray(workspace) ? (
        <p role="alert">
          Workspace must be a mapping. Repair it in block YAML.
        </p>
      ) : (
        <>
          <DraftChoice
            editor={editor}
            label="Workspace mode"
            path={[...path, "mode"]}
            options={["direct", "overlay"]}
          />
          <p>
            Sources use stage input aliases. An empty target mounts at the root;
            directories must not overlap.
          </p>
          {Array.isArray(row.sources) ? (
            row.sources.map((_, index) => (
              <div className="studio-group" key={index}>
                <strong>Source {index + 1}</strong>
                {binding(
                  `Source ${index + 1} artifact`,
                  [...path, "sources", index, "artifact"],
                  required,
                )}
                <DraftField
                  editor={editor}
                  label={`Source ${index + 1} target`}
                  path={[...path, "sources", index, "target"]}
                  placeholder="Workspace root"
                />
                <button
                  className="ui-btn"
                  onClick={() =>
                    onRemove(
                      [...path, "sources", index],
                      `workspace source ${index + 1}`,
                    )
                  }
                >
                  Remove source {index + 1}
                </button>
              </div>
            ))
          ) : (
            <p>Sources must be a list. Repair this value in block YAML.</p>
          )}
          <button
            className="ui-btn"
            disabled={Array.isArray(row.sources) && row.sources.length >= 32}
            onClick={() => {
              try {
                onReplace(
                  appendDraft(draft, [...path, "sources"], {
                    artifact: required[0] ?? "",
                    target: `source-${Array.isArray(row.sources) ? row.sources.length + 1 : 1}`,
                  }),
                );
              } catch {
                edit(() => {
                  throw new Error(
                    "Repair workspace sources in block YAML before adding a source.",
                  );
                });
              }
            }}
          >
            Add workspace source
          </button>
          <h4>Restore state</h4>
          {row.state === undefined ? (
            <button
              className="ui-btn"
              onClick={() =>
                onPatch([...path, "state"], {
                  artifact:
                    slots.find((name) => name.includes("state")) ??
                    slots[0] ??
                    "",
                })
              }
            >
              Add state input
            </button>
          ) : (
            <>
              {binding(
                "State input artifact",
                [...path, "state", "artifact"],
                slots,
              )}
              <button
                className="ui-btn"
                onClick={() =>
                  onRemove([...path, "state"], "workspace state input")
                }
              >
                Remove state input
              </button>
            </>
          )}
          <h4>Export</h4>
          {row.export === undefined ? (
            <button
              className="ui-btn"
              disabled={row.mode !== "overlay"}
              onClick={() =>
                edit((document) => {
                  const available = { ...results };
                  const slot = (base: string, mediaType: string) => {
                    const existing = Object.keys(available).find((name) => {
                      const item = record(available[name]);
                      return (
                        item.from === undefined &&
                        Array.isArray(item.mediaTypes) &&
                        item.mediaTypes.length === 1 &&
                        item.mediaTypes[0] === mediaType
                      );
                    });
                    if (existing) return existing;
                    const name = uniqueName(available, base);
                    const value = { required: true, mediaTypes: [mediaType] };
                    document.setIn(
                      [...stagePath, "result", "artifacts", name],
                      value,
                    );
                    available[name] = value;
                    return name;
                  };
                  document.setIn([...path, "export"], {
                    state: slot("workspace_state", OVERLAY_MEDIA_TYPE),
                    diff: slot("workspace_diff", "text/x-diff"),
                  });
                })
              }
            >
              Add workspace export
            </button>
          ) : (
            <>
              {binding(
                "State output",
                [...path, "export", "state"],
                Object.keys(results),
              )}
              {binding(
                "Diff output",
                [...path, "export", "diff"],
                Object.keys(results),
              )}
              <button
                className="ui-btn"
                onClick={() =>
                  onRemove(
                    [...path, "export"],
                    "workspace export configuration",
                  )
                }
              >
                Remove workspace export
              </button>
            </>
          )}
          {row.mode !== "overlay" ? <p>Export requires overlay mode.</p> : null}
          <button
            className="ui-btn"
            onClick={() => onRemove(path, "workspace configuration")}
          >
            Remove workspace
          </button>
        </>
      )}
      {error ? <p role="alert">{error}</p> : null}
    </details>
  );
}
