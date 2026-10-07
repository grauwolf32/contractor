import { at, record, textValue, type Path } from "./document";
import { useDraftEdit, type FormEditor } from "./form-editor";
import { setSelectionField, setEscalationMode } from "./advanced-edits";
import { CatalogField } from "./catalog-picker";

function SelectionForm({
  editor,
  path,
  title,
  allowClear = false,
}: {
  editor: FormEditor;
  path: Path;
  title: string;
  allowClear?: boolean;
}) {
  const { edit, error } = useDraftEdit(editor.draft, editor.onReplace);
  const value = at(editor.draft.value, path),
    row = record(value);
  const credentialMode =
    row.credential === undefined
      ? "inherit"
      : row.credential === null
        ? "clear"
        : "reference";
  return (
    <div className="studio-group">
      <strong>{title}</strong>
      {value !== undefined &&
      (value === null || typeof value !== "object" || Array.isArray(value)) ? (
        <p role="alert">
          This selection must be a mapping. Repair it in block YAML.
        </p>
      ) : (
        <>
          <p>Unset fields inherit their existing configuration.</p>
          {(["modelPolicy", "llmGateway"] as const).map((field) => (
            <CatalogField
              key={field}
              label={`${title} ${field === "modelPolicy" ? "model policy" : "gateway"}`}
              kind={field === "modelPolicy" ? "model-policies" : "llm-gateways"}
              value={textValue(row[field])}
              onChange={(value) =>
                edit((document) =>
                  setSelectionField(document, path, field, value),
                )
              }
            />
          ))}
          <label className="studio-field">
            <span>{title} credential mode</span>
            <select
              value={credentialMode}
              onChange={(event) =>
                edit((document) => {
                  const mode = event.target.value;
                  if (mode === "reference")
                    document.setIn([...path, "credential"], "");
                  else
                    setSelectionField(
                      document,
                      path,
                      "credential",
                      mode === "clear" ? null : undefined,
                    );
                })
              }
            >
              <option value="inherit">Inherit</option>
              <option value="reference">Credential ID</option>
              {allowClear || credentialMode === "clear" ? (
                <option value="clear" disabled={!allowClear}>
                  Clear credential{!allowClear ? " (escalation only)" : ""}
                </option>
              ) : null}
            </select>
          </label>
          {credentialMode === "reference" ? (
            <label className="studio-field">
              <span>{title} credential ID</span>
              <input
                key={textValue(row.credential)}
                defaultValue={textValue(row.credential)}
                onBlur={(event) => {
                  if (event.target.value !== row.credential)
                    edit((document) =>
                      setSelectionField(
                        document,
                        path,
                        "credential",
                        event.target.value,
                      ),
                    );
                }}
              />
            </label>
          ) : null}
          {value !== undefined ? (
            <button
              className="ui-btn"
              onClick={() =>
                editor.onRemove(path, `${title.toLowerCase()} selection`)
              }
            >
              Remove {title.toLowerCase()} selection
            </button>
          ) : null}
        </>
      )}
      {error ? <p role="alert">{error}</p> : null}
    </div>
  );
}

export function ExecutionDefaultsForm({ editor }: { editor: FormEditor }) {
  const value = at(editor.draft.value, ["spec", "executionConfig"]);
  if (
    value !== undefined &&
    (value === null || typeof value !== "object" || Array.isArray(value))
  )
    return (
      <p role="alert">
        Execution configuration must be a mapping. Repair it in block YAML.
      </p>
    );
  return (
    <>
      <p>
        Defaults apply across this Workflow. Stage selections can override
        individual fields. Planner selections apply to modeled planners.
      </p>
      <SelectionForm
        editor={editor}
        path={["spec", "executionConfig", "planner"]}
        title="Default planner"
      />
      <SelectionForm
        editor={editor}
        path={["spec", "executionConfig", "workers"]}
        title="Default workers"
      />
      {at(editor.draft.value, ["spec", "executionConfig"]) !== undefined ? (
        <button
          className="ui-btn"
          onClick={() =>
            editor.onRemove(
              ["spec", "executionConfig"],
              "all execution selections",
            )
          }
        >
          Remove all execution selections
        </button>
      ) : null}
    </>
  );
}

function StageSelections({
  editor,
  path,
  agents,
  allowClear = false,
  title,
}: {
  editor: FormEditor;
  path: Path;
  agents: string[];
  allowClear?: boolean;
  title: string;
}) {
  const configured = Object.keys(
    record(at(editor.draft.value, [...path, "agents"])),
  );
  return (
    <>
      <SelectionForm
        editor={editor}
        path={[...path, "planner"]}
        title={`${title} planner`}
        allowClear={allowClear}
      />
      {[...new Set([...agents, ...configured])].map((name) => (
        <SelectionForm
          key={name}
          editor={editor}
          path={[...path, "agents", name]}
          title={`${title} agent ${name}`}
          allowClear={allowClear}
        />
      ))}
    </>
  );
}
export function StageExecutionForm({
  editor,
  name,
}: {
  editor: FormEditor;
  name: string;
}) {
  const path = ["spec", "executionConfig", "stages", name];
  const agents = Object.keys(
    record(at(editor.draft.value, ["spec", "stages", name, "agents"])),
  );
  return (
    <details className="studio-settings">
      <summary>Stage execution overrides</summary>
      <StageSelections
        editor={editor}
        path={path}
        agents={agents}
        title="Stage"
      />
      {at(editor.draft.value, path) !== undefined ? (
        <button
          className="ui-btn"
          onClick={() => editor.onRemove(path, "stage execution overrides")}
        >
          Remove stage execution overrides
        </button>
      ) : null}
    </details>
  );
}

export function EscalationForm({
  editor,
  path,
  agents,
  outcome,
}: {
  editor: FormEditor;
  path: Path;
  agents: string[];
  outcome: string;
}) {
  const row = record(at(editor.draft.value, path)),
    mode = row.ref !== undefined ? "reference" : "inline";
  const { edit, error } = useDraftEdit(editor.draft, editor.onReplace);
  return (
    <details className="studio-settings">
      <summary>{outcome} escalation configuration</summary>
      <label className="studio-field">
        <span>{outcome} escalation mode</span>
        <select
          value={mode}
          onChange={(event) =>
            edit((document) =>
              setEscalationMode(document, path, event.target.value),
            )
          }
        >
          <option value="reference">Published configuration</option>
          <option value="inline">Inline selections</option>
        </select>
      </label>
      {mode === "reference" ? (
        <CatalogField
          label={`${outcome} execution configuration`}
          kind="execution-configs"
          value={textValue(row.ref)}
          onChange={(value) => editor.onPatch([...path, "ref"], value)}
        />
      ) : (
        <StageSelections
          editor={editor}
          path={path}
          agents={agents}
          allowClear
          title={`${outcome} escalation`}
        />
      )}
      {error ? <p role="alert">{error}</p> : null}
    </details>
  );
}
