import type { EvalArtifact } from "../../api/evals";
import { artifactDetailPath } from "../../app/artifact-path";

export function artifactHref(a: EvalArtifact): string {
  return artifactDetailPath(
    a.scope === "user" ? { kind: "user" } : { kind: a.scope, id: a.scopeId },
    a,
  );
}
