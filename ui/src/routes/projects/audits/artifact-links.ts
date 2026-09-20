import type { Audit } from "../../../api/audits";

type AuditExactArtifact = Audit["inputs"][string];

export function exactArtifactLink(
  projectId: string,
  artifact: AuditExactArtifact,
) {
  return `/projects/${encodeURIComponent(projectId)}/artifacts/${encodeURIComponent(artifact.ref.namespace)}/${encodeURIComponent(artifact.ref.name)}?revision=${encodeURIComponent(artifact.ref.revision)}`;
}
