import type { ArtifactArchiveScope } from "../../api/artifact-archive";

/** UI path of an Artifact binding's detail page, optionally at a revision. */
export function artifactDetailPath(
  scope: ArtifactArchiveScope,
  ref: { namespace: string; name: string; revision?: string },
  projectRoot: "/projects" | "/evals" = "/projects",
): string {
  const owner =
    scope.kind === "user"
      ? ""
      : `${scope.kind === "run" ? "/runs" : projectRoot}/${encodeURIComponent(scope.id)}`;
  const revision =
    ref.revision === undefined
      ? ""
      : `?revision=${encodeURIComponent(ref.revision)}`;
  return `${owner}/artifacts/${encodeURIComponent(ref.namespace)}/${encodeURIComponent(ref.name)}${revision}`;
}
