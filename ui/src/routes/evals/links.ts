import type { EvalArtifact } from "../../api/evals";
import { artifactDetailPath } from "../artifacts/paths";

export function artifactHref(a: EvalArtifact): string {
  return artifactDetailPath(
    a.scope === "user" ? { kind: "user" } : { kind: a.scope, id: a.scopeId },
    a,
  );
}
