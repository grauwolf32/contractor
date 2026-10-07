import { isMap, isScalar, type Document } from "yaml";
import type { Path } from "./document";

/** Removing the last override restores inheritance rather than emitting {}. */
export function setSelectionField(
  document: Document,
  path: Path,
  field: string,
  value: string | null | undefined,
) {
  if (value === undefined || value === "") {
    document.deleteIn([...path, field]);
    const selection = document.getIn(path, true);
    if (isMap(selection) && selection.items.length === 0)
      document.deleteIn(path);
    const parent = path.slice(0, -1);
    const agents = document.getIn(parent, true);
    if (
      parent.at(-1) === "agents" &&
      isMap(agents) &&
      agents.items.length === 0
    )
      document.deleteIn(parent);
  } else document.setIn([...path, field], value);
}

export function setEscalationMode(
  document: Document,
  path: Path,
  mode: string,
) {
  if (mode === "reference") {
    document.deleteIn([...path, "planner"]);
    document.deleteIn([...path, "agents"]);
    document.setIn([...path, "ref"], "");
  } else {
    document.deleteIn([...path, "ref"]);
    if (
      !document.hasIn([...path, "planner"]) &&
      !document.hasIn([...path, "agents"])
    )
      document.setIn(
        [...path, "planner"],
        document.createNode({ modelPolicy: "" }),
      );
  }
}
export function setArgumentSource(
  document: Document,
  path: Path,
  source: string,
) {
  document.setIn([...path, "source"], source);
  if (source === "literal") {
    document.deleteIn([...path, "name"]);
    if (!document.hasIn([...path, "value"]))
      document.setIn([...path, "value"], "");
  } else {
    document.deleteIn([...path, "value"]);
    if (!document.hasIn([...path, "name"]))
      document.setIn([...path, "name"], "");
  }
}
export function renameArgument(document: Document, path: Path, name: string) {
  if (!/^[A-Za-z_][A-Za-z0-9_]{0,63}$/.test(name))
    throw new Error(
      "Use an argument identifier of up to 64 letters, numbers or underscores, beginning with a letter or underscore.",
    );
  const old = path.at(-1),
    map = document.getIn(path.slice(0, -1), true);
  if (!isMap(map)) throw new Error("Arguments must be a mapping.");
  if (name === old) return;
  if (map.has(name)) throw new Error(`Argument ${name} already exists.`);
  const pair = map.items.find(
    (pair) => isScalar(pair.key) && pair.key.value === old,
  );
  if (pair && isScalar(pair.key)) pair.key.value = name;
}
