import { useLocation, useParams, useSearchParams } from "react-router";
import type { Audit } from "../../../api/audits";
import {
  checksHref,
  parseCheckFilter,
  parseProject,
} from "../../checks/list/filters";
import { sectionPath, type CheckSection } from "./check-links";

/** Full Check detail has two entries, with one shared section navigation. */
export function useCheckNavigation(audit: Audit) {
  const location = useLocation();
  const path = useParams();
  const [search] = useSearchParams();
  const inChecks = location.pathname === "/checks";
  const context = {
    filter: inChecks ? parseCheckFilter(search.get("state")) : ("all" as const),
    projectId: inChecks ? parseProject(search.get("project")) : audit.projectId,
  };
  const listHref = checksHref({ ...context, checkId: undefined });
  function sectionHref(section: CheckSection, extra?: Record<string, string>) {
    if (!inChecks) {
      const query = new URLSearchParams(extra);
      const root = sectionPath(audit.projectId, audit.auditId, section);
      return query.size ? `${root}?${query}` : root;
    }
    const selected = checksHref({ ...context, checkId: audit.auditId });
    const query = new URLSearchParams(selected.split("?")[1]);
    if (section !== "overview") query.set("section", section);
    for (const [key, value] of Object.entries(extra ?? {}))
      query.set(key, value);
    return `/checks?${query}`;
  }
  return {
    section: (inChecks ? search.get("section") : path.section) ?? "overview",
    sectionHref,
    listHref,
  };
}
