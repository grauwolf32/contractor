import { useId, useState } from "react";
import { Document, parseDocument } from "yaml";
import { Dialog, DialogHeader } from "../../app/dialog";
import { Button, Field } from "../../ui";
import { DraftChoice, DraftField, DraftList } from "./advanced-fields";
import {
  at,
  record,
  textValue,
  uniqueName,
  type Draft,
  type Path,
} from "./document";
import { nodeValue, type StudioNode } from "./graph";
import { OUTCOMES } from "./validation";
import { CatalogField, type CatalogKind } from "./catalog-picker";
import { WorkspaceForm } from "./workspace-form";
import {
  EscalationForm,
  ExecutionDefaultsForm,
  StageExecutionForm,
} from "./execution-form";
import { ToolExecutionForm } from "./tool-execution-form";
import type { FormEditor } from "./form-editor";

export function Inspector({
  draft,
  node,
  onPatch,
  onReplace,
  onRename,
  onRemove,
  onConnect,
}: {
  draft: Draft;
  node: StudioNode | undefined;
  onPatch: (path: Path, value: unknown) => void;
  onReplace: (draft: Draft) => void;
  onRename: (name: string) => void;
  onRemove: (node: StudioNode) => void;
  onConnect: (outcome: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  if (!node)
    return (
      <aside className="studio-inspector">
        <h2>Properties</h2>
        <p>Select a block on the graph.</p>
      </aside>
    );
  const value = record(nodeValue(draft, node)),
    p = node.path;
  const editor: FormEditor = {
    draft,
    onPatch,
    onReplace,
    onRemove: (path, title) => onRemove({ ...node, path, title }),
  };
  const field = (
    label: string,
    path: Path,
    multiline = false,
    number = false,
  ) => (
    <DraftField
      key={JSON.stringify(path)}
      editor={editor}
      label={label}
      path={path}
      multiline={multiline}
      number={number}
    />
  );
  const select = (
    label: string,
    path: Path,
    options: string[],
    fallback = "",
  ) => (
    <DraftChoice
      editor={editor}
      label={label}
      path={path}
      options={options}
      fallback={fallback}
    />
  );
  const catalogField = (label: string, path: Path, kind: CatalogKind) => (
    <CatalogField
      key={JSON.stringify(path)}
      label={label}
      kind={kind}
      value={textValue(at(draft.value, path))}
      onChange={(value) => onPatch(path, value)}
    />
  );
  const check = (label: string, path: Path) => (
    <label className="studio-checkbox">
      <input
        type="checkbox"
        checked={at(draft.value, path) === true}
        onChange={(event) => onPatch(path, event.target.checked)}
      />
      {label}
    </label>
  );
  const remove = (path: Path, title: string) => (
    <Button onClick={() => onRemove({ ...node, path, title })}>
      Remove {title}
    </Button>
  );
  const media = (path: Path) => (
    <DraftList
      editor={editor}
      label="Media types (comma separated)"
      path={path}
    />
  );
  return (
    <aside className="studio-inspector" aria-label="Block properties">
      <span className="studio-eyebrow">{node.type}</span>
      <h2>{node.title}</h2>
      {node.removable && typeof p[2] === "string" ? (
        <Field label="Block name" className="studio-field">
          {(control) => (
            <input
              {...control}
              key={node.title}
              defaultValue={node.title}
              onBlur={(event) => {
                if (event.target.value !== node.title)
                  onRename(event.target.value);
              }}
            />
          )}
        </Field>
      ) : null}
      {node.type === "failure" ? (
        <p>
          Failed and interrupted branches arrive here after their retry or
          escalation limits. Edit those branches on the stage.
        </p>
      ) : (
        <>
          {node.type === "stage" ? (
            <>
              {field("Objective", [...p, "objective"], true)}
              {field("Planner", [...p, "planner"])}
              {select(
                "Session",
                [...p, "session"],
                ["isolated", "shared"],
                "isolated",
              )}
              {field("Instructions file", [...p, "instructions", "ref"])}
              <Button
                onClick={() => onPatch(["spec", "entryStage"], node.title)}
                disabled={record(draft.value.spec).entryStage === node.title}
              >
                Use as entry stage
              </Button>
              <h3>Agents</h3>
              {Object.keys(record(value.agents)).map((name) => (
                <div key={name} className="studio-group">
                  <strong>{name}</strong>
                  {catalogField(
                    "Template",
                    [...p, "agents", name, "template"],
                    "agent-templates",
                  )}
                  {field("Namespace", [...p, "agents", name, "namespace"])}
                  {remove([...p, "agents", name], `agent ${name}`)}
                </div>
              ))}
              <Button
                onClick={() => {
                  const name = uniqueName(value.agents, "agent");
                  onPatch([...p, "agents", name], {
                    template: "worker@1",
                    namespace: name,
                  });
                }}
              >
                Add agent
              </Button>
              <h3>Outcome transitions</h3>
              {OUTCOMES.map((outcome) => {
                const path = [...p, "on", outcome],
                  action = record(at(draft.value, path)),
                  type = Object.keys(action)[0] ?? "";
                const options =
                  outcome === "succeeded"
                    ? ["next", "succeed"]
                    : ["next", "retry", "escalate", "fail"];
                return (
                  <div className="studio-group" key={outcome}>
                    <Field label={outcome} className="studio-field">
                      {(control) => (
                        <select
                          {...control}
                          value={type}
                          onChange={(event) => {
                            const next = event.target.value;
                            const first =
                              Object.keys(
                                record(record(draft.value.spec).stages),
                              ).find((name) => name !== node.title) ?? "";
                            onPatch(
                              path,
                              next === "next"
                                ? { next: first }
                                : next === "retry" || next === "escalate"
                                  ? {
                                      [next]: {
                                        maxAttempts: next === "retry" ? 2 : 1,
                                        ...(next === "escalate"
                                          ? {
                                              executionConfig: {
                                                ref: "execution@1",
                                              },
                                            }
                                          : {}),
                                        then: { fail: {} },
                                      },
                                    }
                                  : { [next]: {} },
                            );
                          }}
                        >
                          {!options.includes(type) ? (
                            <option value="">Choose…</option>
                          ) : null}
                          {options.map((option) => (
                            <option key={option}>{option}</option>
                          ))}
                        </select>
                      )}
                    </Field>
                    {type === "next"
                      ? select(
                          "Destination stage",
                          [...path, "next"],
                          Object.keys(record(record(draft.value.spec).stages)),
                        )
                      : null}
                    {type === "retry" || type === "escalate" ? (
                      <>
                        {field(
                          "Maximum attempts",
                          [...path, type, "maxAttempts"],
                          false,
                          true,
                        )}
                        {type === "escalate" ? (
                          <EscalationForm
                            editor={editor}
                            path={[...path, type, "executionConfig"]}
                            agents={Object.keys(record(value.agents))}
                            outcome={outcome}
                          />
                        ) : null}
                        <Field label="After attempts" className="studio-field">
                          {(control) => (
                            <select
                              {...control}
                              value={
                                textValue(
                                  record(record(action[type]).then).next,
                                ) || "__fail"
                              }
                              onChange={(event) =>
                                onPatch(
                                  [...path, type, "then"],
                                  event.target.value === "__fail"
                                    ? { fail: {} }
                                    : { next: event.target.value },
                                )
                              }
                            >
                              <option value="__fail">Fail</option>
                              {Object.keys(
                                record(record(draft.value.spec).stages),
                              ).map((name) => (
                                <option key={name}>{name}</option>
                              ))}
                            </select>
                          )}
                        </Field>
                      </>
                    ) : null}
                    <Button onClick={() => onConnect(outcome)}>
                      Connect {outcome} on canvas
                    </Button>
                  </div>
                );
              })}
              <h3>Incoming files</h3>
              {Object.entries(record(record(value.context).artifacts)).map(
                ([name, wire]) => (
                  <div className="studio-group" key={name}>
                    <strong>{name}</strong>
                    {field("Namespace", [
                      ...p,
                      "context",
                      "artifacts",
                      name,
                      "namespace",
                    ])}
                    {field("File name", [
                      ...p,
                      "context",
                      "artifacts",
                      name,
                      "name",
                    ])}
                    {check("Required file", [
                      ...p,
                      "context",
                      "artifacts",
                      name,
                      "required",
                    ])}
                    <small>
                      {textValue(record(wire).namespace)}/
                      {textValue(record(wire).name)}
                    </small>
                    {remove(
                      [...p, "context", "artifacts", name],
                      `wire ${name}`,
                    )}
                  </div>
                ),
              )}
              <Button
                onClick={() => {
                  const name = uniqueName(
                    record(value.context).artifacts,
                    "file",
                  );
                  onPatch([...p, "context", "artifacts", name], {
                    namespace: "inputs",
                    name:
                      Object.keys(record(record(draft.value.spec).inputs))[0] ??
                      "source",
                    required: true,
                  });
                }}
              >
                Add incoming file
              </Button>
              <h3>Result files</h3>
              {Object.entries(record(record(value.result).artifacts)).map(
                ([name, slot]) => (
                  <div className="studio-group" key={name}>
                    <strong>{name}</strong>
                    {check("Required result", [
                      ...p,
                      "result",
                      "artifacts",
                      name,
                      "required",
                    ])}
                    {media([...p, "result", "artifacts", name, "mediaTypes"])}
                    {record(slot).from ? (
                      <>
                        {field("Producer namespace", [
                          ...p,
                          "result",
                          "artifacts",
                          name,
                          "from",
                          "namespace",
                        ])}
                        {field("Producer file", [
                          ...p,
                          "result",
                          "artifacts",
                          name,
                          "from",
                          "name",
                        ])}
                      </>
                    ) : (
                      <p>Runtime workspace export; producer binding omitted.</p>
                    )}
                    {remove(
                      [...p, "result", "artifacts", name],
                      `result ${name}`,
                    )}
                  </div>
                ),
              )}
              <Button
                onClick={() => {
                  const name = uniqueName(
                      record(value.result).artifacts,
                      "result",
                    ),
                    agent = Object.entries(record(value.agents))[0];
                  onPatch([...p, "result", "artifacts", name], {
                    required: true,
                    mediaTypes: ["application/yaml"],
                    from: {
                      namespace:
                        textValue(record(agent?.[1]).namespace) ||
                        agent?.[0] ||
                        "worker",
                      name,
                    },
                  });
                }}
              >
                Add result file
              </Button>
              <h3>Workflow output bindings</h3>
              {Object.entries(record(value.workflowOutputs)).map(([name]) => (
                <div className="studio-group" key={name}>
                  {select(
                    name,
                    [...p, "workflowOutputs", name],
                    Object.keys(record(record(value.result).artifacts)),
                  )}
                  {remove(
                    [...p, "workflowOutputs", name],
                    `output binding ${name}`,
                  )}
                </div>
              ))}
              <Field label="Bind workflow output" className="studio-field">
                {(control) => (
                  <select
                    {...control}
                    value=""
                    onChange={(event) =>
                      onPatch(
                        [...p, "workflowOutputs", event.target.value],
                        Object.keys(
                          record(record(value.result).artifacts),
                        )[0] ?? "",
                      )
                    }
                  >
                    <option value="">Choose an output…</option>
                    {Object.keys(record(record(draft.value.spec).outputs))
                      .filter(
                        (name) =>
                          !Object.hasOwn(record(value.workflowOutputs), name),
                      )
                      .map((name) => (
                        <option key={name}>{name}</option>
                      ))}
                  </select>
                )}
              </Field>
              <WorkspaceForm editor={editor} stagePath={p} />
              <StageExecutionForm editor={editor} name={node.title} />
            </>
          ) : null}
          {node.type === "input" || node.type === "output" ? (
            <>
              {check("Required", [...p, "required"])}
              {node.type === "output"
                ? check("Primary output", [...p, "primary"])
                : null}
              {media([...p, "mediaTypes"])}
            </>
          ) : null}
          {node.type === "parameter"
            ? check("Required", [...p, "required"])
            : null}
          {node.type === "routing" ? (
            <ExecutionDefaultsForm editor={editor} />
          ) : null}
          {node.type === "toolExecution" ? (
            <ToolExecutionForm editor={editor} />
          ) : null}
          {node.type === "role" ? (
            <>
              {select(
                "Role kind",
                [...p, "kind"],
                ["prepare", "discovery", "check", "assessment"],
              )}
              {catalogField("Workflow", [...p, "ref"], "workflows")}
              {field(
                "Maximum run attempts",
                [...p, "maxRunAttempts"],
                false,
                true,
              )}
              <h3>Input bindings</h3>
              {Object.entries(record(value.inputs)).map(([name, input]) => (
                <div className="studio-group" key={name}>
                  <strong>{name}</strong>
                  <Field label="Source" className="studio-field">
                    {(control) => (
                      <select
                        {...control}
                        value={textValue(record(input).source)}
                        onChange={(event) => {
                          const path = [...p, "inputs", name],
                            source = event.target.value;
                          const document = draft.document.clone();
                          document.setIn([...path, "source"], source);
                          if (
                            ["item-package", "execution-manifest"].includes(
                              source,
                            )
                          )
                            document.deleteIn([...path, "name"]);
                          if (
                            !["prepare-output", "retained-output"].includes(
                              source,
                            )
                          )
                            document.deleteIn([...path, "role"]);
                          onReplace({ ...draft, source: document.toString() });
                        }}
                      >
                        {[
                          "audit-input",
                          "item-package",
                          "execution-manifest",
                          "prepare-output",
                          "retained-output",
                        ].map((source) => (
                          <option key={source}>{source}</option>
                        ))}
                      </select>
                    )}
                  </Field>
                  {[
                    "audit-input",
                    "prepare-output",
                    "retained-output",
                  ].includes(textValue(record(input).source))
                    ? field("Source name", [...p, "inputs", name, "name"])
                    : null}
                  {["prepare-output", "retained-output"].includes(
                    textValue(record(input).source),
                  )
                    ? select(
                        "Producer role",
                        [...p, "inputs", name, "role"],
                        Object.keys(record(record(draft.value.spec).workflows)),
                      )
                    : null}
                  {remove([...p, "inputs", name], `binding ${name}`)}
                </div>
              ))}
              <Button
                onClick={() => {
                  const name = uniqueName(value.inputs, "input");
                  onPatch([...p, "inputs", name], {
                    source: "audit-input",
                    name:
                      Object.keys(record(record(draft.value.spec).inputs))[0] ??
                      "source",
                  });
                }}
              >
                Add input binding
              </Button>
              <p>
                Edit input sources, output names, parameters and worker
                completion in block YAML.
              </p>
            </>
          ) : null}
          {node.type === "inventory" ? (
            <>
              {field("Implementation", [...p, "implementation"])}
              {select(
                "Item workflow role",
                [...p, "itemWorkflowRole"],
                Object.keys(record(record(draft.value.spec).workflows)),
              )}
              {field("Source input", [...p, "source", "name"])}
            </>
          ) : null}
          {node.type === "execution" ? (
            <>
              {[
                "maxRounds",
                "batchSize",
                "maxItemsPerRound",
                "maxItemsTotal",
                "maxSubmittedRuns",
                "maxItemRunAttempts",
                "deadlineSeconds",
                "maxEvidenceBytes",
              ].map((key) => field(key, [...p, key], false, true))}
              {select(
                "Incomplete round",
                [...p, "incompleteRound"],
                ["assess-with-gaps", "fail"],
              )}
            </>
          ) : null}
          {node.type === "interaction" ? (
            <>
              {Object.entries({
                activeChecks: ["prohibited", "automatic", "approval-required"],
                findingConfirmation: ["human-required", "disabled"],
                notApplicable: ["human-required", "profile-rule"],
                reportAcceptance: ["automatic", "human-required"],
              }).map(([key, options]) => (
                <div key={key}>{select(key, [...p, key], options)}</div>
              ))}
            </>
          ) : null}
          {node.type === "agent" ? (
            <>
              {field("Name", ["metadata", "name"])}
              {field("Version", ["metadata", "version"])}
              {field("Description", ["spec", "description"], true)}
              {field("Runtime", ["spec", "runtime"])}
            </>
          ) : null}
          {node.type === "modelPolicy"
            ? catalogField(node.title, p, "model-policies")
            : null}
          {node.type === "sandboxProfile" ? field(node.title, p) : null}
          {node.type === "instructions"
            ? field("Instructions file", [...p, "ref"])
            : null}
          {node.type === "summarizer" ? (
            <>
              {catalogField(
                "Model policy",
                [...p, "modelPolicy"],
                "model-policies",
              )}
              {field("Instructions file", [...p, "instructions", "ref"])}
              {field(
                "Context window ratio",
                [...p, "contextWindowRatio"],
                false,
                true,
              )}
            </>
          ) : null}
          {node.type === "toolset" ? (
            <>
              {field("Toolset", [...p, "ref"])}
              <DraftList
                editor={editor}
                label="Tools (comma separated)"
                path={[...p, "tools"]}
              />
            </>
          ) : null}
          {node.type === "skill" ? (
            <>
              {field("Skill name", [...p, "name"])}
              {field("Namespace", [...p, "namespace"])}
            </>
          ) : null}
          <Button onClick={() => setEditing(true)}>Edit block YAML</Button>
          {node.removable ? (
            <Button variant="danger" onClick={() => onRemove(node)}>
              Remove block
            </Button>
          ) : null}
        </>
      )}
      {editing ? (
        <BlockYAML
          draft={draft}
          node={node}
          onClose={() => setEditing(false)}
          onReplace={onReplace}
        />
      ) : null}
    </aside>
  );
}

function BlockYAML({
  draft,
  node,
  onClose,
  onReplace,
}: {
  draft: Draft;
  node: StudioNode;
  onClose: () => void;
  onReplace: (draft: Draft) => void;
}) {
  const id = useId();
  const [initial] = useState(() => {
    const document = new Document();
    document.contents = draft.document.getIn(
      node.path,
      true,
    ) as typeof document.contents;
    return document.toString();
  });
  const [source, setSource] = useState(initial),
    [error, setError] = useState(""),
    [discard, setDiscard] = useState(false);
  return (
    <Dialog
      className="ui-dialog ui-panel"
      labelledBy={id}
      onRequestClose={() => (source === initial ? onClose() : setDiscard(true))}
    >
      <DialogHeader
        id={id}
        title={`Edit ${node.title} YAML`}
        close={{
          label: "Close block YAML",
          onClose: () => (source === initial ? onClose() : setDiscard(true)),
        }}
      />
      <p>
        Changes apply only to this block. Other fields and comments stay in the
        document.
      </p>
      <Field label="Block YAML" className="studio-field">
        {(control) => (
          <textarea
            {...control}
            className="studio-source"
            rows={18}
            value={source}
            onChange={(event) => {
              setSource(event.target.value);
              setError("");
              setDiscard(false);
            }}
          />
        )}
      </Field>
      {error ? <p role="alert">{error}</p> : null}
      {discard ? (
        <p role="alert">
          Discard unapplied block edits?{" "}
          <Button onClick={onClose}>Discard block edits</Button>
          <Button onClick={() => setDiscard(false)}>Keep editing</Button>
        </p>
      ) : null}
      <div className="ui-dialog-actions">
        <Button
          onClick={() => {
            try {
              if (source !== initial) {
                const block = parseDocument(source, { uniqueKeys: true });
                if (block.errors.length)
                  throw new Error(block.errors[0]!.message);
                const document = draft.document.clone();
                document.setIn(node.path, block.contents);
                // Parent checks enforce the same size/alias bounds as file imports.
                onReplace({ ...draft, source: document.toString() });
              }
              onClose();
            } catch (error) {
              setError(
                error instanceof Error ? error.message : "Invalid block YAML.",
              );
            }
          }}
        >
          Apply block YAML
        </Button>
      </div>
    </Dialog>
  );
}
