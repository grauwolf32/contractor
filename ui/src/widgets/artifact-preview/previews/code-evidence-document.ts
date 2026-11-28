import { parseDocument } from "yaml";
import {
  ARTIFACT_NAME_PATTERN,
  ARTIFACT_REVISION_PATTERN,
  MAXIMUM_PREVIEW_BYTES,
} from "../../../api/artifacts";
import { DIGEST } from "../../../api/contract";
import { sourceLines } from "../source-lines";

interface Location {
  file: string;
  startLine: number;
  endLine: number;
  digest: string;
}
export interface CodeEvidenceDocument {
  schema: "contractor.trace-finding-evidence.v1";
  source: Location & {
    artifact: { namespace: string; name: string; revision: string };
  };
  workspace: Location & {
    snippetStartLine: number;
    snippetEndLine: number;
    text: string;
  };
  requestDigest: string;
}

function object(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
function keys(value: Record<string, unknown>, names: string[]) {
  return (
    Object.keys(value).length === names.length &&
    names.every((name) => Object.hasOwn(value, name))
  );
}
function line(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 1;
}
function location(value: unknown): value is Location & Record<string, unknown> {
  return (
    object(value) &&
    typeof value.file === "string" &&
    value.file.length > 0 &&
    line(value.startLine) &&
    line(value.endLine) &&
    value.startLine <= value.endLine &&
    typeof value.digest === "string" &&
    DIGEST.test(value.digest)
  );
}

/** Display admission only. Runtime owns source validation and coordinate translation (FND-R231). */
export function parseCodeEvidence(
  source: string,
): CodeEvidenceDocument | undefined {
  if (new TextEncoder().encode(source).length > MAXIMUM_PREVIEW_BYTES)
    return undefined;
  try {
    const value: unknown = JSON.parse(source);
    if (
      !object(value) ||
      !keys(value, ["schema", "source", "workspace", "requestDigest"]) ||
      value.schema !== "contractor.trace-finding-evidence.v1" ||
      typeof value.requestDigest !== "string" ||
      !DIGEST.test(value.requestDigest) ||
      !location(value.source) ||
      !location(value.workspace)
    )
      return undefined;
    const original = value.source,
      workspace = value.workspace;
    if (
      !keys(original, ["file", "startLine", "endLine", "digest", "artifact"]) ||
      !keys(workspace, [
        "file",
        "startLine",
        "endLine",
        "digest",
        "snippetStartLine",
        "snippetEndLine",
        "text",
      ])
    )
      return undefined;
    const artifact = original.artifact;
    if (
      !object(artifact) ||
      !keys(artifact, ["namespace", "name", "revision"]) ||
      typeof artifact.namespace !== "string" ||
      !ARTIFACT_NAME_PATTERN.test(artifact.namespace) ||
      typeof artifact.name !== "string" ||
      !ARTIFACT_NAME_PATTERN.test(artifact.name) ||
      typeof artifact.revision !== "string" ||
      !ARTIFACT_REVISION_PATTERN.test(artifact.revision)
    )
      return undefined;
    // LIM-FND-55: present only a complete bounded snippet with coherent saved coordinates.
    if (
      !line(workspace.snippetStartLine) ||
      !line(workspace.snippetEndLine) ||
      workspace.snippetStartLine > workspace.startLine ||
      workspace.endLine > workspace.snippetEndLine ||
      workspace.snippetEndLine - workspace.snippetStartLine >= 400 ||
      typeof workspace.text !== "string" ||
      new TextEncoder().encode(workspace.text).length > 128 * 1024 ||
      sourceLines(workspace.text, 400)?.length !==
        workspace.snippetEndLine - workspace.snippetStartLine + 1
    )
      return undefined;
    if (
      parseDocument(source, { uniqueKeys: true, prettyErrors: false }).errors
        .length
    )
      return undefined;
    return value as unknown as CodeEvidenceDocument;
  } catch {
    return undefined;
  }
}
