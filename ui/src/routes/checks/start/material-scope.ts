import { useQuery } from "@tanstack/react-query";
import { dereference, validate, type Schema } from "@cfworker/json-schema";
import engagementSchema from "../../../../../api/v1alpha1/engagement.schema.json" with { type: "json" };
import type { ArtifactMetadata } from "../../../api/artifacts";
import type { AuditProfile } from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { previewProjectArtifact } from "../../../api/project-artifacts";
import { queryKeys } from "../../../api/query-keys";
import { AUTOMATIC_DOCUMENT_BYTES } from "../../../app/document-limits";
import type { ChecklistDraft } from "./checklist-state";
import { CHECKLIST_SCHEMA } from "./checklist-document";
import { scopeSummary } from "./check-types";
import type { InputSlot } from "./use-start-check";

const ENGAGEMENT_MEDIA_TYPE = "application/vnd.contractor.engagement+json";
const engagementContract = engagementSchema as unknown as Schema;
const engagementReferences = dereference(engagementContract);

interface EngagementScope {
  schema: "contractor.engagement.v1";
  accessMode: "source-assisted" | "blackbox";
  target: { origins: string[] };
  scope: {
    routes: {
      origin: string;
      pathPrefixes: string[];
      methods: string[];
      addressRanges: string[];
    }[];
    exclusions: {
      origin: string;
      pathPrefixes: string[];
      methods: string[];
      addressRanges: string[];
    }[];
  };
  identities: { id: string; kind: string; role?: string; tenant?: string }[];
  budgets: {
    activeSeconds: number;
    rounds: number;
    submittedRuns: number;
    evidenceBytes: number;
  };
}

type ScopeContent =
  | {
      kind: "api";
      operations: { method: string; path: string }[];
      partial: boolean;
    }
  | { kind: "checklist"; count: number }
  | { kind: "engagement"; document: EngagementScope };

function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** Presentation of exact content, never inventory admission or target authorization. */
async function readScopeContent(
  kind: "api" | "checklist" | "engagement",
  source: string,
): Promise<ScopeContent | undefined> {
  if (kind === "api") {
    const { readOpenApiEndpoints } =
      await import("../../artifacts/previews/openapi-document");
    const endpoints = readOpenApiEndpoints(source);
    return endpoints === undefined ? undefined : { kind: "api", ...endpoints };
  }
  const { parse } = await import("yaml");
  let value: unknown;
  try {
    value = parse(source, { uniqueKeys: true, maxAliasCount: 0 });
  } catch {
    return undefined;
  }
  if (kind === "engagement") {
    return validate(
      value,
      engagementContract,
      "2020-12",
      engagementReferences,
      true,
    ).valid
      ? { kind: "engagement", document: value as EngagementScope }
      : undefined;
  }
  if (
    !record(value) ||
    value.schema !== CHECKLIST_SCHEMA ||
    !Array.isArray(value.items)
  )
    return undefined;
  if (
    !value.items.every(
      (item) =>
        record(item) &&
        typeof item.key === "string" &&
        typeof item.statement === "string",
    )
  )
    return undefined;
  return { kind: "checklist", count: value.items.length };
}

function useSelectedContent(
  projectId: string,
  kind: "api" | "checklist" | "engagement" | undefined,
  material: ArtifactMetadata | undefined,
) {
  const api = usePublicAPI();
  const eligible =
    material !== undefined &&
    material.size <= AUTOMATIC_DOCUMENT_BYTES &&
    kind !== undefined;
  const identity = material?.artifact;
  const query = useQuery({
    queryKey: [
      ...queryKeys.projects.artifacts.all(projectId),
      "scope",
      kind,
      identity?.namespace,
      identity?.name,
      identity?.revision,
    ],
    enabled: eligible,
    queryFn: async ({ signal }) => {
      if (!eligible)
        throw new Error("Choose a small material to inspect its scope.");
      const source = await previewProjectArtifact(
        api,
        projectId,
        material,
        signal,
      );
      return (await readScopeContent(kind, source)) ?? null;
    },
    retry: false,
    staleTime: Infinity,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  });
  return { ...query, eligible, material, loading: eligible && query.isPending };
}

export function isEngagementSlot(slot: InputSlot): boolean {
  return slot.input.mediaTypes.includes(ENGAGEMENT_MEDIA_TYPE);
}

export function useMaterialScope(
  projectId: string,
  profile: AuditProfile | undefined,
  slots: readonly InputSlot[],
  checklist: ChecklistDraft | undefined,
) {
  const inventory = profile?.inventory;
  const kind =
    inventory?.implementation === "checklist@1"
      ? "checklist"
      : inventory?.implementation === "openapi-operations@1" ||
          inventory?.implementation === "openapi-scans@1"
        ? "api"
        : undefined;
  const sourceName =
    inventory?.source?.source === "audit-input"
      ? inventory.source.name
      : undefined;
  const source = slots.find((slot) => slot.name === sourceName);
  const writing = checklist?.mode === "write";
  const content = useSelectedContent(
    projectId,
    writing ? undefined : kind,
    writing ? undefined : source?.selected,
  );
  const engagementSlots = slots.filter(isEngagementSlot);
  const engagement = useSelectedContent(
    projectId,
    "engagement",
    engagementSlots[0]?.selected,
  );
  const base = profile === undefined ? undefined : scopeSummary(profile);
  let summary = base?.size ?? "Choose a check type";
  if (
    inventory?.implementation === "standard-mappings@1" &&
    profile?.standards.length === 0
  )
    summary = "Whole source archive";
  if (writing)
    summary =
      checklist.applied === undefined
        ? "Checklist draft"
        : `${checklist.applied.items.length} ${checklist.applied.items.length === 1 ? "item" : "items"}`;
  else if (content.data?.kind === "api" && !content.data.partial)
    summary = `${content.data.operations.length} endpoints in the API spec`;
  else if (content.data?.kind === "checklist")
    summary = `${content.data.count} ${content.data.count === 1 ? "item" : "items"} in the checklist`;
  const showInventory =
    kind !== undefined && (!writing || checklist.applied !== undefined);
  return {
    summary,
    base,
    kind,
    source,
    content,
    engagement,
    engagementSlots,
    showInventory,
    show: showInventory || engagementSlots.length > 0,
    writing,
  };
}

export type MaterialScope = ReturnType<typeof useMaterialScope>;
