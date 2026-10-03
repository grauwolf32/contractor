import type { Audit } from "../../../api/audits";
import { artifactDetailPath } from "../../artifacts/paths";

type AuditExactArtifact = Audit["inputs"][string];

export function exactArtifactLink(
  projectId: string,
  artifact: AuditExactArtifact,
) {
  return artifactDetailPath({ kind: "project", id: projectId }, artifact.ref);
}
