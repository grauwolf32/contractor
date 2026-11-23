import { Button, Field } from "../../ui";
import { at, record, textValue, uniqueName } from "./document";
import { DraftField } from "./advanced-fields";
import { useDraftEdit, type FormEditor } from "./form-editor";
import { renameArgument, setArgumentSource } from "./advanced-edits";

export function ToolExecutionForm({ editor }: { editor: FormEditor }) {
  const path = ["spec", "execution"],
    row = record(at(editor.draft.value, path));
  const { edit, error } = useDraftEdit(editor.draft, editor.onReplace);
  const spec = record(editor.draft.value.spec),
    firstTool = Array.isArray(spec.toolsets)
      ? record(spec.toolsets[0]).tools
      : undefined;
  return (
    <>
      {at(editor.draft.value, path) === undefined ? (
        <Button
          onClick={() =>
            editor.onPatch(path, {
              tool: Array.isArray(firstTool) ? (firstTool[0] ?? "") : "",
              arguments: {},
              resultArtifact: "report",
              timeoutSeconds: 300,
            })
          }
        >
          Configure tool execution
        </Button>
      ) : (
        <>
          <DraftField
            editor={editor}
            label="Execution tool"
            path={[...path, "tool"]}
          />
          <DraftField
            editor={editor}
            label="Result artifact"
            path={[...path, "resultArtifact"]}
          />
          <DraftField
            editor={editor}
            label="Timeout seconds"
            path={[...path, "timeoutSeconds"]}
            number
          />
          <h3>Arguments</h3>
          {Object.entries(record(row.arguments)).map(([name, value]) => {
            const p = [...path, "arguments", name],
              binding = record(value),
              literalType = typeof binding.value;
            return (
              <div className="studio-group" key={name}>
                <Field label="Argument name" className="studio-field">
                  {(control) => (
                    <input
                      {...control}
                      defaultValue={name}
                      onBlur={(event) => {
                        if (event.target.value !== name)
                          edit((document) =>
                            renameArgument(document, p, event.target.value),
                          );
                      }}
                    />
                  )}
                </Field>
                <Field label={`${name} source`} className="studio-field">
                  {(control) => (
                    <select
                      {...control}
                      value={textValue(binding.source)}
                      onChange={(event) =>
                        edit((document) =>
                          setArgumentSource(document, p, event.target.value),
                        )
                      }
                    >
                      {!["parameter", "artifact", "literal"].includes(
                        textValue(binding.source),
                      ) ? (
                        <option value={textValue(binding.source)}>
                          Choose…
                        </option>
                      ) : null}
                      {["parameter", "artifact", "literal"].map((source) => (
                        <option key={source}>{source}</option>
                      ))}
                    </select>
                  )}
                </Field>
                {binding.source === "literal" ? (
                  <>
                    <Field
                      label={`${name} literal type`}
                      className="studio-field"
                    >
                      {(control) => (
                        <select
                          {...control}
                          value={
                            ["string", "number", "boolean"].includes(
                              literalType,
                            )
                              ? literalType
                              : "unsupported"
                          }
                          onChange={(event) =>
                            editor.onPatch(
                              [...p, "value"],
                              event.target.value === "boolean"
                                ? false
                                : event.target.value === "number"
                                  ? 0
                                  : "",
                            )
                          }
                        >
                          {!["string", "number", "boolean"].includes(
                            literalType,
                          ) ? (
                            <option value="unsupported">
                              Unsupported value
                            </option>
                          ) : null}
                          <option value="string">String</option>
                          <option value="number">Number</option>
                          <option value="boolean">Boolean</option>
                        </select>
                      )}
                    </Field>
                    {literalType === "boolean" ? (
                      <label className="studio-checkbox">
                        <input
                          type="checkbox"
                          checked={binding.value === true}
                          onChange={(event) =>
                            editor.onPatch(
                              [...p, "value"],
                              event.target.checked,
                            )
                          }
                        />
                        {name} value
                      </label>
                    ) : ["string", "number"].includes(literalType) ? (
                      <DraftField
                        editor={editor}
                        label={`${name} value`}
                        path={[...p, "value"]}
                        number={literalType === "number"}
                      />
                    ) : (
                      <p>
                        Choose a scalar type or repair the literal in block
                        YAML.
                      </p>
                    )}
                  </>
                ) : (
                  <DraftField
                    editor={editor}
                    label={`${name} binding name`}
                    path={[...p, "name"]}
                  />
                )}
                <Button onClick={() => editor.onRemove(p, `argument ${name}`)}>
                  Remove argument {name}
                </Button>
              </div>
            );
          })}
          <Button
            disabled={Object.keys(record(row.arguments)).length >= 32}
            onClick={() =>
              editor.onPatch(
                [...path, "arguments", uniqueName(row.arguments, "argument")],
                { source: "literal", value: "" },
              )
            }
          >
            Add argument
          </Button>
          <Button
            onClick={() =>
              editor.onRemove(path, "tool execution configuration")
            }
          >
            Remove tool execution configuration
          </Button>
        </>
      )}
      {error ? <p role="alert">{error}</p> : null}
    </>
  );
}
