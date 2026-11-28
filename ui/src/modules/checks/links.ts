export const CHECK_SECTIONS = [
  ["overview", "Overview"],
  ["work", "Work"],
  ["hypotheses", "Hypotheses"],
  ["findings", "Possible issues"],
  ["materials", "Materials"],
  ["activity", "Activity"],
] as const;
export type CheckSection = (typeof CHECK_SECTIONS)[number][0];

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
