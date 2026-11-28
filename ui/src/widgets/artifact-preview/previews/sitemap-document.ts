import { parseDocument } from "yaml";
import type { components } from "../../../api/generated/public";
import { conformsTo } from "../../../api/contract";
import { MAXIMUM_PREVIEW_BYTES } from "../../../api/artifacts";

export type SitemapDocument = components["schemas"]["SitemapDocument"];

/** Present the persisted public contract; do not reconstruct domain identities. */
export function parseSitemapDocument(
  source: string,
): SitemapDocument | undefined {
  if (new TextEncoder().encode(source).length > MAXIMUM_PREVIEW_BYTES)
    return undefined;
  try {
    const value: unknown = JSON.parse(source);
    // JSON syntax alone silently accepts repeated members. Match the existing
    // HTTP material reader's duplicate-key check before interpreting content.
    if (
      parseDocument(source, { uniqueKeys: true, prettyErrors: false }).errors
        .length
    )
      return undefined;
    if (!conformsTo("SitemapDocument", value)) return undefined;
    const document = value as SitemapDocument;
    // Ambiguous identities cannot support truthful selection. Identity
    // calculation and the other domain invariants remain with the publisher.
    const groups = new Set(document.groups.map((group) => group.id));
    const members = document.groups.flatMap((group) => group.members);
    if (
      groups.size !== document.groups.length ||
      new Set(members.map((member) => member.id)).size !== members.length ||
      new Set(document.sources.map((source) => source.provenance.sourceId))
        .size !== document.sources.length
    )
      return undefined;
    return document;
  } catch {
    return undefined;
  }
}
