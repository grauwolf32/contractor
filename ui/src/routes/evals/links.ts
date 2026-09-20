import type { EvalArtifact } from "../../api/evals";

export function artifactHref(a: EvalArtifact): string {
  const scope =
    a.scope === "user"
      ? ""
      : `/${a.scope === "run" ? "runs" : "projects"}/${encodeURIComponent(a.scopeId)}`;
  return `${scope}/artifacts/${encodeURIComponent(a.namespace)}/${encodeURIComponent(a.name)}?revision=${encodeURIComponent(a.revision)}`;
}
