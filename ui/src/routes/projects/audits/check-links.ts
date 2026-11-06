/** Sections of the check page, the last path segment of its URL. */
type CheckSection =
  "overview" | "work" | "hypotheses" | "findings" | "materials" | "activity";

/** /projects/:projectId/audits/:auditId */
export function checkPath(projectId: string, auditId: string): string {
  return `/projects/${encodeURIComponent(projectId)}/audits/${encodeURIComponent(auditId)}`;
}

export function sectionPath(
  projectId: string,
  auditId: string,
  section: CheckSection,
): string {
  const root = checkPath(projectId, auditId);
  return section === "overview" ? root : `${root}/${section}`;
}
