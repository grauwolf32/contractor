import type { To } from "react-router";

import type { Group } from "./assessments";

/** Sections of the check page, the last path segment of its URL. */
export type CheckSection =
  "overview" | "coverage" | "findings" | "reviews" | "runs" | "report";

export const CHECK_SECTIONS: readonly CheckSection[] = [
  "overview",
  "coverage",
  "findings",
  "reviews",
  "runs",
  "report",
];

export function parseSection(value: string | undefined): CheckSection {
  return CHECK_SECTIONS.find((section) => section === value) ?? "overview";
}

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

const ITEM_HASH = "#check-";

/** `#check-<itemId>`: selects an item on the coverage section. */
export function itemHash(itemId: string): string {
  return `${ITEM_HASH}${encodeURIComponent(itemId)}`;
}

/** The item a `#check-<itemId>` hash selects. */
export function hashedItem(hash: string): string | undefined {
  if (!hash.startsWith(ITEM_HASH)) return undefined;
  try {
    const itemId = decodeURIComponent(hash.slice(ITEM_HASH.length));
    return itemId === "" ? undefined : itemId;
  } catch {
    return undefined;
  }
}

/** Parameters of the item list that travel with every list link. */
const LIST_PARAMS = ["result", "q"] as const;

/** Sections where `auditRevision` pins the item list (elsewhere it pins a queue). */
export function pinsItemList(section: CheckSection): boolean {
  return section === "overview" || section === "coverage";
}

function search(params: URLSearchParams): string {
  const value = params.toString();
  return value === "" ? "" : `?${value}`;
}

/**
 * Links of one check page. The list filters (`result`, `q`) travel with the
 * list links and the section tabs; the revision pin travels only between
 * the overview and the item list, where it pins the list.
 */
export interface CheckLinks {
  /** The overview: All activity and the technical details. */
  overview: To;
  /** The item list without a selection. */
  list: To;
  item: (itemId: string) => To;
  /** The item list filtered to a result group, optionally pinned. */
  group: (group: Group, revision?: number) => To;
  section: (section: CheckSection) => To;
  /** A section with its own parameters, keeping the list filters. */
  deep: (section: CheckSection, params: Record<string, string>) => To;
}

export function checkLinks(
  projectId: string,
  auditId: string,
  current: URLSearchParams,
  section: CheckSection,
): CheckLinks {
  const listParams = new URLSearchParams();
  for (const key of LIST_PARAMS) {
    const value = current.get(key);
    if (value !== null && value !== "") listParams.set(key, value);
  }
  const pinned = new URLSearchParams(listParams);
  const pin = current.get("auditRevision");
  if (pinsItemList(section) && pin !== null) pinned.set("auditRevision", pin);
  const path = (target: CheckSection) =>
    sectionPath(projectId, auditId, target);
  return {
    overview: { pathname: path("overview"), search: search(pinned) },
    list: { pathname: path("coverage"), search: search(pinned) },
    item: (itemId) => ({
      pathname: path("coverage"),
      search: search(pinned),
      hash: itemHash(itemId),
    }),
    group: (group, revision) => {
      const next = new URLSearchParams(listParams);
      if (group === "all") next.delete("result");
      else next.set("result", group);
      if (revision !== undefined) next.set("auditRevision", String(revision));
      return { pathname: path("coverage"), search: search(next) };
    },
    section: (target) => ({
      pathname: path(target),
      search: search(pinsItemList(target) ? pinned : listParams),
    }),
    deep: (target, params) => {
      const next = new URLSearchParams(listParams);
      for (const [key, value] of Object.entries(params)) next.set(key, value);
      return { pathname: path(target), search: search(next) };
    },
  };
}
