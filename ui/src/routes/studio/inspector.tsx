import { useId, useState } from "react";
import { Document, parseDocument } from "yaml";
import { Dialog, DialogHeader } from "../../app/dialog";
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
  const field = (
    label: string,
    path: Path,
    multiline = false,
    number = false,
  ) => {
    const current = textValue(at(draft.value, path));
    const commit = (value: string) => {
      if (value !== current)
        onPatch(path, number ? (value === "" ? null : Number(value)) : value);
    };
    return (
      <label className="studio-field" key={JSON.stringify(path)}>
        <span>{label}</span>
        {multiline ? (
          <textarea
            key={current}
            defaultValue={current}
            rows={4}
            onBlur={(event) => commit(event.target.value)}
          />
        ) : (
          <input
            key={current}
            defaultValue={current}
            type={number ? "number" : "text"}
            onBlur={(event) => commit(event.target.value)}
          />
        )}
      </label>
    );
  };
  const select = (
    label: string,
    path: Path,
    options: string[],
    fallback = "",
  ) => {
    const current = textValue(at(draft.value, path)) || fallback;
    return (
      <label className="studio-field">
        <span>{label}</span>
        <select
          value={current}
          onChange={(event) => onPatch(path, event.target.value)}
        >
          {!options.includes(current) ? (
            <option value={current}>{current || "Choose…"}</option>
          ) : null}
          {options.map((option) => (
            <option key={option}>{option}</option>
          ))}
        </select>
      </label>
    );
  };
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
    <button
      className="ui-btn"
      onClick={() => onRemove({ ...node, path, title })}
    >
      Remove {title}
    </button>
  );
  const media = (path: Path) => {
    const types = at(draft.value, path);
    return (
      <label className="studio-field">
        <span>Media types (comma separated)</span>
        <input
          key={JSON.stringify(types)}
          defaultValue={Array.isArray(types) ? types.join(", ") : ""}
          onBlur={(event) => {
            const next = event.target.value
              .split(",")
              .map((type) => type.trim())
              .filter(Boolean);
            if (JSON.stringify(next) !== JSON.stringify(types))
              onPatch(path, next);
          }}
        />
      </label>
    );
  };
  return (
    <aside className="studio-inspector" aria-label="Block properties">
      <span className="studio-eyebrow">{node.type}</span>
      <h2>{node.title}</h2>
      {node.removable && typeof p[2] === "string" ? (
        <label className="studio-field">
          <span>Block name</span>
          <input
            key={node.title}
            defaultValue={node.title}
            onBlur={(event) => {
              if (event.target.value !== node.title)
                onRename(event.target.value);
            }}
          />
        </label>
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
              <button
                className="ui-btn"
                onClick={() => onPatch(["spec", "entryStage"], node.title)}
                disabled={record(draft.value.spec).entryStage === node.title}
              >
                Use as entry stage
              </button>
              <h3>Agents</h3>
              {Object.keys(record(value.agents)).map((name) => (
                <div key={name} className="studio-group">
                  <strong>{name}</strong>
                  {field("Template", [...p, "agents", name, "template"])}
                  {field("Namespace", [...p, "agents", name, "namespace"])}
                  {remove([...p, "agents", name], `agent ${name}`)}
                </div>
              ))}
              <button
                className="ui-btn"
                onClick={() => {
                  const name = uniqueName(value.agents, "agent");
                  onPatch([...p, "agents", name], {
                    template: "worker@1",
                    namespace: name,
                  });
                }}
              >
                Add agent
              </button>
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
                    <label className="studio-field">
                      <span>{outcome}</span>
                      <select
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
                    </label>
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
                        {type === "escalate"
                          ? field("Escalation configuration", [
                              ...path,
                              type,
                              "executionConfig",
                              "ref",
                            ])
                          : null}
                        <label className="studio-field">
                          <span>After attempts</span>
                          <select
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
                        </label>
                      </>
                    ) : null}
                    <button
                      className="ui-btn"
                      onClick={() => onConnect(outcome)}
                    >
                      Connect {outcome} on canvas
                    </button>
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
              <button
                className="ui-btn"
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
              </button>
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
              <button
                className="ui-btn"
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
              </button>
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
              <label className="studio-field">
                <span>Bind workflow output</span>
                <select
                  value=""
                  onChange={(event) =>
                    onPatch(
                      [...p, "workflowOutputs", event.target.value],
                      Object.keys(record(record(value.result).artifacts))[0] ??
                        "",
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
              </label>
              <p>Workspace configuration remains available in block YAML.</p>
            </>
          ) : null}
          {node.type === "input" || node.type === "output" ? (
            <>
              {check("Required", [...p, "required"])}
              {node.type === "output"
                ? check("Primary output", [...p, "primary"])
                : null}
              <label className="studio-field">
                <span>Media types (comma separated)</span>
                <input
                  key={JSON.stringify(value.mediaTypes)}
                  defaultValue={
                    Array.isArray(value.mediaTypes)
                      ? value.mediaTypes.join(", ")
                      : ""
                  }
                  onBlur={(event) => {
                    const types = event.target.value
                      .split(",")
                      .map((type) => type.trim())
                      .filter(Boolean);
                    if (
                      JSON.stringify(types) !== JSON.stringify(value.mediaTypes)
                    )
                      onPatch([...p, "mediaTypes"], types);
                  }}
                />
              </label>
            </>
          ) : null}
          {node.type === "parameter"
            ? check("Required", [...p, "required"])
            : null}
          {node.type === "role" ? (
            <>
              {select(
                "Role kind",
                [...p, "kind"],
                ["prepare", "discovery", "check", "assessment"],
              )}
              {field("Workflow", [...p, "ref"])}
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
                  <label className="studio-field">
                    <span>Source</span>
                    <select
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
                  </label>
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
              <button
                className="ui-btn"
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
              </button>
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
          {["modelPolicy", "sandboxProfile"].includes(node.type)
            ? field(node.title, p)
            : null}
          {node.type === "instructions"
            ? field("Instructions file", [...p, "ref"])
            : null}
          {node.type === "summarizer" ? (
            <>
              {field("Model policy", [...p, "modelPolicy"])}
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
              <label className="studio-field">
                <span>Tools (comma separated)</span>
                <input
                  key={JSON.stringify(value.tools)}
                  defaultValue={
                    Array.isArray(value.tools) ? value.tools.join(", ") : ""
                  }
                  onBlur={(event) =>
                    onPatch(
                      [...p, "tools"],
                      event.target.value
                        .split(",")
                        .map((tool) => tool.trim())
                        .filter(Boolean),
                    )
                  }
                />
              </label>
            </>
          ) : null}
          {node.type === "skill" ? (
            <>
              {field("Skill name", [...p, "name"])}
              {field("Namespace", [...p, "namespace"])}
            </>
          ) : null}
          <button className="ui-btn" onClick={() => setEditing(true)}>
            Edit block YAML
          </button>
          {node.removable ? (
            <button
              className="ui-btn"
              data-variant="danger"
              onClick={() => onRemove(node)}
            >
              Remove block
            </button>
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
      className="project-dialog studio-dialog"
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
      <label className="studio-field">
        <span>Block YAML</span>
        <textarea
          className="studio-source"
          rows={18}
          value={source}
          onChange={(event) => {
            setSource(event.target.value);
            setError("");
            setDiscard(false);
          }}
        />
      </label>
      {error ? <p role="alert">{error}</p> : null}
      {discard ? (
        <p role="alert">
          Discard unapplied block edits?{" "}
          <button className="ui-btn" onClick={onClose}>
            Discard block edits
          </button>
          <button className="ui-btn" onClick={() => setDiscard(false)}>
            Keep editing
          </button>
        </p>
      ) : null}
      <button
        className="ui-btn"
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
      </button>
    </Dialog>
  );
}
