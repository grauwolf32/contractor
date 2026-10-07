import { useEffect, useId, useMemo, useState } from "react";
import { Link, useBlocker, useSearchParams } from "react-router";
import { Dialog, DialogHeader } from "../../app/dialog";
import { useDocumentTitle } from "../../app/document-title";
import { saveBlob } from "../../app/download";
import { StudioCanvas } from "./canvas";
import {
  addBlock,
  at,
  KINDS,
  kindLabel,
  MAX_SOURCE_BYTES,
  patchDraft,
  readDraft,
  removeBlock,
  renameBlock,
  sourceDiff,
  starter,
  textValue,
  type BlockType,
  type Draft,
  type Path,
} from "./document";
import { buildGraph, pathID, type StudioNode } from "./graph";
import { Inspector } from "./inspector";
import { StudioLive } from "./live";
import { validateDraft } from "./validation";
import "./studio.css";

interface EditorState {
  generation: number;
  draft: Draft;
  baseline: string;
  saved: string;
  buffer: string;
  past: string[];
  future: string[];
}
const initialState = (draft: Draft, generation = 0): EditorState => ({
  generation,
  draft,
  baseline: draft.source,
  saved: draft.source,
  buffer: draft.source,
  past: [],
  future: [],
});
function boundedHistory(sources: string[]): string[] {
  let bytes = 0;
  return sources
    .slice(-50)
    .reverse()
    .filter((source) => {
      bytes += source.length * 2;
      return bytes <= 8 * MAX_SOURCE_BYTES;
    })
    .reverse();
}

export function StudioRoute() {
  useDocumentTitle("Node Studio · Library");
  const [params] = useSearchParams();
  const [state, setState] = useState(() =>
    initialState(
      starter(KINDS.find((kind) => params.get("kind") === kind) ?? "Workflow"),
    ),
  );
  const { draft } = state;
  const [selected, setSelected] = useState("");
  const [mode, setMode] = useState(params.has("audit") ? "Live" : "Design");
  const [pane, setPane] = useState("Graph"),
    [tab, setTab] = useState("Problems");
  const [importing, setImporting] = useState(false),
    [replacement, setReplacement] = useState<Draft>();
  const [removing, setRemoving] = useState<StudioNode>();
  const [connect, setConnect] = useState<{ path: Path; outcome: string }>();
  const [notice, setNotice] = useState("");
  const dirty = draft.source !== state.saved || state.buffer !== draft.source;
  const pendingYAML = state.buffer !== draft.source;
  const blocker = useBlocker(dirty);
  const graph = useMemo(() => buildGraph(draft), [draft]);
  const problems = useMemo(() => validateDraft(draft), [draft]);
  const node =
    graph.nodes.find((node) => node.id === selected) ??
    graph.nodes.find((node) => node.type !== "failure");
  const errors = problems.filter(
    (problem) => problem.severity === "error",
  ).length;
  const warnings = problems.length - errors;
  const headingId = useId();
  useEffect(() => {
    if (!dirty) return;
    const beforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", beforeUnload);
    return () => window.removeEventListener("beforeunload", beforeUnload);
  }, [dirty]);
  useEffect(() => {
    if (!connect) return;
    const cancel = (event: KeyboardEvent) => {
      if (event.key === "Escape") setConnect(undefined);
    };
    window.addEventListener("keydown", cancel);
    return () => window.removeEventListener("keydown", cancel);
  }, [connect]);
  const push = (next: Draft) => {
    const checked = readDraft(next.source);
    if (checked.source === draft.source) return;
    setState((current) => ({
      ...current,
      draft: checked,
      buffer: checked.source,
      past: boundedHistory([...current.past, current.draft.source]),
      future: [],
    }));
    setNotice("");
  };
  const patch = (path: Path, value: unknown) => {
    try {
      push(patchDraft(draft, path, value));
    } catch (error) {
      setNotice(
        error instanceof Error ? error.message : "Could not edit this field.",
      );
    }
  };
  const select = (node: StudioNode, showProperties = true) => {
    setSelected(node.id);
    if (showProperties) setPane("Properties");
  };
  const add = (type: BlockType) => {
    if (pendingYAML) {
      setNotice("Apply or revert the YAML edits before changing the graph.");
      return;
    }
    try {
      const next = addBlock(draft, type);
      push(next.draft);
      setSelected(pathID(next.path));
      setPane("Properties");
    } catch (error) {
      setNotice(
        error instanceof Error ? error.message : "Could not add a block.",
      );
    }
  };
  const replace = (next: Draft) => {
    if (dirty) setReplacement(next);
    else {
      setState((current) => initialState(next, current.generation + 1));
      setSelected("");
      setConnect(undefined);
      setTab("Problems");
      setMode("Design");
    }
  };
  const exportYAML = () => {
    try {
      const source = state.buffer;
      const exported = readDraft(source);
      const name =
        textValue(at(exported.value, ["metadata", "name"])).replace(
          /[^A-Za-z0-9_.-]/g,
          "_",
        ) || "draft";
      saveBlob(
        new Blob([source], { type: "application/yaml;charset=utf-8" }),
        `${name}.yaml`,
      );
      setState((current) => ({
        ...current,
        saved: source,
        draft: exported,
        buffer: source,
        past:
          source === current.draft.source
            ? current.past
            : boundedHistory([...current.past, current.draft.source]),
        future: source === current.draft.source ? current.future : [],
      }));
      setNotice(
        "YAML exported. Validate the complete configuration bundle before use.",
      );
    } catch (error) {
      setNotice(
        error instanceof Error ? error.message : "YAML could not be exported.",
      );
    }
  };
  const blocks: [BlockType, string][] =
    draft.kind === "Workflow"
      ? [
          ["stage", "Stage"],
          ["input", "Input"],
          ["output", "Output"],
          ["parameter", "Parameter"],
        ]
      : draft.kind === "AuditProfile"
        ? [
            ["input", "Input"],
            ["role", "Workflow role"],
          ]
        : [
            ["toolset", "Toolset"],
            ["skill", "Skill"],
          ];
  const diff = sourceDiff(state.baseline, draft.source);
  return (
    <div className="studio-page">
      <header className="studio-header">
        <div>
          <Link to="/catalog/workflows">Library</Link>
          <h1>Node Studio</h1>
          <p>Local YAML draft · {dirty ? "Unsaved changes" : "In memory"}</p>
        </div>
        <div className="studio-header-actions">
          <label className="studio-kind">
            <span>Definition</span>
            <select
              value={draft.kind}
              onChange={(event) =>
                replace(starter(event.target.value as Draft["kind"]))
              }
            >
              {KINDS.map((kind) => (
                <option key={kind} value={kind}>
                  {kindLabel(kind)}
                </option>
              ))}
            </select>
          </label>
          <button
            className="ui-btn"
            onClick={() => replace(starter(draft.kind))}
          >
            New draft
          </button>
          <button className="ui-btn" onClick={() => setImporting(true)}>
            Import YAML
          </button>
          <button
            className="ui-btn"
            onClick={() => {
              setPane("Console");
              if (pendingYAML) {
                setTab("YAML");
                try {
                  const pendingProblems = validateDraft(
                    readDraft(state.buffer),
                  );
                  const pendingErrors = pendingProblems.filter(
                    (problem) => problem.severity === "error",
                  ).length;
                  setNotice(
                    `Pending YAML: ${pendingErrors} errors, ${pendingProblems.length - pendingErrors} warnings. Apply YAML to update the graph and Problems.`,
                  );
                } catch (error) {
                  setNotice(
                    error instanceof Error ? error.message : "Invalid YAML.",
                  );
                }
                return;
              }
              setTab("Problems");
              setNotice(
                `${errors} errors, ${warnings} warnings in local checks. Bundle references require CLI validation.`,
              );
            }}
          >
            Validate
          </button>
          <button
            className="ui-btn"
            data-variant="primary"
            onClick={exportYAML}
          >
            Export YAML
          </button>
        </div>
      </header>
      <div className="studio-toolbar">
        <div role="group" aria-label="Studio mode">
          {["Design", "Live"].map((value) => (
            <button
              key={value}
              className="ui-btn"
              aria-pressed={mode === value}
              onClick={() => setMode(value)}
            >
              {value}
            </button>
          ))}
        </div>
        <span className="studio-identity">
          {textValue(at(draft.value, ["metadata", "name"]))}@
          {textValue(at(draft.value, ["metadata", "version"]))}
        </span>
        <button
          className="ui-btn"
          disabled={!state.past.length || pendingYAML}
          onClick={() => {
            const source = state.past.at(-1)!;
            setState((current) => ({
              ...current,
              draft: readDraft(source),
              buffer: source,
              past: current.past.slice(0, -1),
              future: [current.draft.source, ...current.future],
            }));
          }}
        >
          Undo
        </button>
        <button
          className="ui-btn"
          disabled={!state.future.length || pendingYAML}
          onClick={() => {
            const source = state.future[0]!;
            setState((current) => ({
              ...current,
              draft: readDraft(source),
              buffer: source,
              past: boundedHistory([...current.past, current.draft.source]),
              future: current.future.slice(1),
            }));
          }}
        >
          Redo
        </button>
        <span>
          {errors} errors · {warnings} warnings
        </span>
      </div>
      {notice ? (
        <p className="studio-notice" role="status">
          {notice}
        </p>
      ) : null}
      {mode === "Live" ? (
        <StudioLive initialAudit={params.get("audit") ?? ""} draft={draft} />
      ) : (
        <>
          <div
            className="studio-mobile-tabs"
            role="group"
            aria-label="Studio panels"
          >
            {["Graph", "Blocks", "Properties", "Console"].map((value) => (
              <button
                className="ui-btn"
                key={value}
                aria-pressed={pane === value}
                onClick={() => setPane(value)}
              >
                {value}
              </button>
            ))}
          </div>
          <div className="studio-workspace" data-pane={pane}>
            <aside className="studio-palette">
              <h2>Blocks</h2>
              <p>Add or drag onto the canvas.</p>
              {blocks.map(([type, label]) => (
                <button
                  className="studio-palette-block"
                  disabled={pendingYAML}
                  key={type}
                  draggable={!pendingYAML}
                  onDragStart={(event) =>
                    event.dataTransfer.setData(
                      "application/contractor-studio-block",
                      type,
                    )
                  }
                  onClick={() => add(type)}
                >
                  + {label}
                </button>
              ))}
              <h3>Definition</h3>
              <label className="studio-field">
                <span>Name</span>
                <input
                  disabled={pendingYAML}
                  key={textValue(at(draft.value, ["metadata", "name"]))}
                  defaultValue={textValue(
                    at(draft.value, ["metadata", "name"]),
                  )}
                  onBlur={(event) => {
                    if (
                      event.target.value !==
                      at(draft.value, ["metadata", "name"])
                    )
                      patch(["metadata", "name"], event.target.value);
                  }}
                />
              </label>
              <label className="studio-field">
                <span>Version</span>
                <input
                  disabled={pendingYAML}
                  key={textValue(at(draft.value, ["metadata", "version"]))}
                  defaultValue={textValue(
                    at(draft.value, ["metadata", "version"]),
                  )}
                  onBlur={(event) => {
                    if (
                      event.target.value !==
                      at(draft.value, ["metadata", "version"])
                    )
                      patch(["metadata", "version"], event.target.value);
                  }}
                />
              </label>
              <p>All other authored settings remain available in YAML.</p>
            </aside>
            <div className="studio-graph-panel">
              <StudioCanvas
                key={state.generation}
                graph={graph}
                selected={node?.id ?? ""}
                onSelect={select}
                problems={problems}
                onAdd={add}
                connect={connect ? JSON.stringify(connect) : undefined}
                onConnect={(target) => {
                  if (!connect || pendingYAML) return;
                  patch([...connect.path, "on", connect.outcome], {
                    next: target.title,
                  });
                  setConnect(undefined);
                  setNotice(
                    "Connection added. Local checks have been updated.",
                  );
                }}
              />
              {connect ? (
                <button
                  className="ui-btn studio-cancel-connect"
                  onClick={() => setConnect(undefined)}
                >
                  Cancel connection
                </button>
              ) : null}
            </div>
            <fieldset
              className="studio-properties-panel"
              disabled={pendingYAML}
            >
              <Inspector
                key={node?.id}
                draft={draft}
                node={node}
                onPatch={patch}
                onReplace={push}
                onRename={(name) => {
                  if (!node) return;
                  try {
                    push(renameBlock(draft, node.path, name));
                    setSelected(pathID([...node.path.slice(0, -1), name]));
                  } catch (error) {
                    setNotice(
                      error instanceof Error
                        ? error.message
                        : "Could not rename this block.",
                    );
                  }
                }}
                onRemove={setRemoving}
                onConnect={(outcome) => {
                  if (node) setConnect({ path: node.path, outcome });
                  setPane("Graph");
                }}
              />
            </fieldset>
            <section className="studio-console" aria-label="Draft console">
              <div
                className="studio-console-tabs"
                role="group"
                aria-label="Console views"
              >
                {["Problems", "YAML", "Diff"].map((value) => (
                  <button
                    className="ui-btn"
                    key={value}
                    aria-pressed={tab === value}
                    onClick={() => setTab(value)}
                  >
                    {value}
                    {value === "Problems" ? ` (${problems.length})` : ""}
                  </button>
                ))}
              </div>
              {pendingYAML ? (
                <p role="status">
                  YAML has unapplied changes. Apply or revert them to edit graph
                  properties.
                </p>
              ) : null}
              {tab === "Problems" ? (
                <div className="studio-console-body">
                  <p>
                    Local structural and graph checks. Resolve installed
                    selectors, media compatibility, instruction files and
                    runtime capabilities with the configuration CLI before use.
                  </p>
                  {problems.length ? (
                    <ul>
                      {problems.map((problem, index) => (
                        <li key={index} data-severity={problem.severity}>
                          <button
                            onClick={() => {
                              const target = graph.nodes
                                .filter((node) =>
                                  node.path.every(
                                    (part, index) =>
                                      problem.path[index] === part,
                                  ),
                                )
                                .sort(
                                  (a, b) => b.path.length - a.path.length,
                                )[0];
                              if (target) select(target);
                              else {
                                setTab("YAML");
                                setPane("Console");
                              }
                            }}
                          >
                            <strong>{problem.severity}</strong> ·{" "}
                            {problem.path.join(".")}: {problem.message}
                          </button>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p>No errors in local checks.</p>
                  )}
                </div>
              ) : null}
              {tab === "YAML" ? (
                <div className="studio-console-body">
                  <label className="studio-field">
                    <span>Authored YAML</span>
                    <textarea
                      className="studio-source"
                      rows={12}
                      value={state.buffer}
                      spellCheck={false}
                      onChange={(event) =>
                        setState((current) => ({
                          ...current,
                          buffer: event.target.value,
                        }))
                      }
                    />
                  </label>
                  <button
                    className="ui-btn"
                    disabled={!pendingYAML}
                    onClick={() => {
                      try {
                        const next = readDraft(state.buffer);
                        if (next.kind !== draft.kind) {
                          setNotice(
                            "Import a different definition kind using Import YAML.",
                          );
                          return;
                        }
                        push(next);
                        setNotice("YAML applied to the graph.");
                      } catch (error) {
                        setNotice(
                          error instanceof Error
                            ? error.message
                            : "Invalid YAML.",
                        );
                      }
                    }}
                  >
                    Apply YAML
                  </button>{" "}
                  <button
                    className="ui-btn"
                    disabled={!pendingYAML}
                    onClick={() =>
                      setState((current) => ({
                        ...current,
                        buffer: current.draft.source,
                      }))
                    }
                  >
                    Revert YAML edits
                  </button>
                </div>
              ) : null}
              {tab === "Diff" ? (
                <div className="studio-console-body">
                  <p>
                    Changes since import or New draft.
                    {pendingYAML
                      ? " Apply YAML to include pending edits in this diff."
                      : ""}
                  </p>
                  {diff.length ? (
                    <pre className="studio-diff" aria-label="YAML diff">
                      {diff.slice(0, 1000).map((line, index) => (
                        <span key={index} data-change={line.kind}>
                          {line.kind === "add"
                            ? "+"
                            : line.kind === "remove"
                              ? "−"
                              : " "}{" "}
                          {line.line}
                          {"\n"}
                        </span>
                      ))}
                    </pre>
                  ) : (
                    <p>No changes.</p>
                  )}
                  {diff.length > 1000 ? (
                    <p>
                      Showing the first 1,000 diff lines. Export YAML for the
                      full document.
                    </p>
                  ) : null}
                </div>
              ) : null}
            </section>
          </div>
        </>
      )}
      {importing ? (
        <ImportYAML
          onClose={() => setImporting(false)}
          onImport={(draft) => {
            setImporting(false);
            replace(draft);
          }}
        />
      ) : null}
      {replacement || removing || blocker.state === "blocked" ? (
        <Dialog
          className="project-dialog studio-dialog"
          labelledBy={headingId}
          onRequestClose={() => {
            setReplacement(undefined);
            setRemoving(undefined);
            if (blocker.state === "blocked") blocker.reset();
          }}
        >
          <DialogHeader
            id={headingId}
            title={
              removing
                ? `Remove ${removing.title}?`
                : "Discard current draft changes?"
            }
          />
          <p>
            {removing
              ? "References to this block will remain and appear in Problems. You can undo the removal."
              : "This draft lives only in this tab. Export YAML to keep it before continuing."}
          </p>
          {!removing ? (
            <button className="ui-btn" onClick={exportYAML}>
              Export YAML
            </button>
          ) : null}
          <button
            className="ui-btn"
            onClick={() => {
              setReplacement(undefined);
              setRemoving(undefined);
              if (blocker.state === "blocked") blocker.reset();
            }}
          >
            Keep editing
          </button>
          <button
            className="ui-btn"
            data-variant="danger"
            onClick={() => {
              if (removing) {
                try {
                  push(removeBlock(draft, removing.path));
                  setSelected("");
                } catch (error) {
                  setNotice(
                    error instanceof Error
                      ? error.message
                      : "Could not remove this block.",
                  );
                }
                setRemoving(undefined);
              } else if (replacement) {
                setState((current) =>
                  initialState(replacement, current.generation + 1),
                );
                setSelected("");
                setConnect(undefined);
                setReplacement(undefined);
                setMode("Design");
                setTab("Problems");
              } else if (blocker.state === "blocked") blocker.proceed();
            }}
          >
            {removing ? "Remove block" : "Discard changes and continue"}
          </button>
        </Dialog>
      ) : null}
    </div>
  );
}

function ImportYAML({
  onImport,
  onClose,
}: {
  onImport: (draft: Draft) => void;
  onClose: () => void;
}) {
  const id = useId(),
    [source, setSource] = useState(""),
    [reading, setReading] = useState(false),
    [error, setError] = useState(""),
    [discard, setDiscard] = useState(false);
  const close = () => (source ? setDiscard(true) : onClose());
  return (
    <Dialog
      className="project-dialog studio-dialog"
      labelledBy={id}
      onRequestClose={close}
    >
      <DialogHeader
        id={id}
        title="Import authored YAML"
        close={{ label: "Close import", onClose: close }}
      />
      <p>
        Open one Workflow, AuditProfile or AgentTemplate file (up to 1 MiB), or
        paste YAML. Your draft stays in this tab; export it to keep your work.
      </p>
      <label className="studio-field">
        <span>YAML file</span>
        <input
          type="file"
          disabled={reading}
          accept=".yaml,.yml,text/yaml,application/yaml"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (!file) return;
            if (file.size > MAX_SOURCE_BYTES) {
              setError("YAML must be at most 1 MiB.");
              return;
            }
            setReading(true);
            void file
              .text()
              .then((value) => {
                setSource(value);
                setError("");
                setDiscard(false);
              })
              .catch(() => setError("The file could not be read."))
              .finally(() => setReading(false));
          }}
        />
      </label>
      <label className="studio-field">
        <span>Paste YAML</span>
        <textarea
          className="studio-source"
          rows={16}
          disabled={reading}
          value={source}
          spellCheck={false}
          onChange={(event) => {
            setSource(event.target.value);
            setError("");
            setDiscard(false);
          }}
        />
      </label>
      {error ? <p role="alert">{error}</p> : null}
      {reading ? <p role="status">Reading YAML file…</p> : null}
      {discard ? (
        <p role="alert">
          Discard pasted YAML?{" "}
          <button className="ui-btn" onClick={onClose}>
            Discard import
          </button>
          <button className="ui-btn" onClick={() => setDiscard(false)}>
            Keep importing
          </button>
        </p>
      ) : null}
      <button
        className="ui-btn"
        data-variant="primary"
        disabled={reading || !source.trim()}
        onClick={() => {
          try {
            onImport(readDraft(source));
          } catch (error) {
            setError(error instanceof Error ? error.message : "Invalid YAML.");
          }
        }}
      >
        Load draft
      </button>
    </Dialog>
  );
}
