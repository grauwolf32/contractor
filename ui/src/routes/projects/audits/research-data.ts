import type { StatusTone } from "../../../app/status-tone";
import { useQuery } from "@tanstack/react-query";
import {
  auditPollInterval,
  getAuditProfile,
  type Audit,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { invalidAPIResponse, requireData } from "../../../api/error";
import type { components } from "../../../api/generated/public";
import { queryKeys } from "../../../api/query-keys";
import { useURLCursorStack } from "../../../app/pagination";
import { pollInterval } from "../../../app/polling";
import { useAuditProjectionRefresh } from "./projection-refresh";

export type ResearchTask = components["schemas"]["AuditTask"];
export type ResearchHypothesis = components["schemas"]["AuditHypothesis"];
export type ResearchMaterial = components["schemas"]["AuditMaterialEntry"];

/** A mutable catalog version cannot silently substitute a different Program pin. */
export function useResearchProgram(audit: Audit) {
  const api = usePublicAPI();
  return useQuery({
    queryKey: [
      ...queryKeys.audits.detail(audit.auditId),
      "program",
      audit.profile.digest,
    ],
    queryFn: async ({ signal }) => {
      const profile = await getAuditProfile(
        api,
        audit.profile.name,
        audit.profile.version,
        signal,
      );
      if (profile.ref.digest !== audit.profile.digest || !profile.program)
        throw new Error(
          "The pinned program is unavailable or its published definition changed. Accepted work retains its original configuration.",
        );
      return { ...profile, program: profile.program };
    },
    staleTime: Infinity,
  });
}

export function useResearchTasks(audit: Audit, hypothesis?: string) {
  const api = usePublicAPI();
  const pages = useURLCursorStack({
    param: hypothesis ? "historyCursor" : "tasksCursor",
    clearOnChange: hypothesis ? ["historyTask", "task"] : ["task"],
  });
  const queryKey = [
    ...queryKeys.audits.detail(audit.auditId),
    "tasks",
    hypothesis,
    pages.cursor,
  ];
  const query = useQuery({
    queryKey,
    queryFn: async ({ signal }) => {
      const data = requireData(
        await api.request((client) =>
          client.GET("/v1/audits/{auditId}/tasks", {
            signal,
            params: {
              path: { auditId: audit.auditId },
              query: {
                limit: 50,
                ...(pages.cursor ? { cursor: pages.cursor } : {}),
                ...(hypothesis ? { hypothesis_id: hypothesis } : {}),
              },
            },
          }),
        ),
      );
      if (
        data.page.hasMore &&
        (data.items.length === 0 ||
          !data.page.nextCursor ||
          data.page.nextCursor === pages.cursor)
      )
        throw invalidAPIResponse(200, "Invalid Task continuation");
      return data;
    },
    refetchInterval: pollInterval<components["schemas"]["AuditTaskPage"]>({
      interval: 5000,
      settled: () => auditPollInterval([audit]) === false,
    }),
  });
  useAuditProjectionRefresh(audit, queryKey);
  return { query, pages };
}

export function useResearchHypotheses(audit: Audit, enabled = true) {
  const api = usePublicAPI();
  const pages = useURLCursorStack({
    param: "hypothesesCursor",
    clearOnChange: ["hypothesis", "historyTask", "historyCursor"],
  });
  const queryKey = ["audit-hypotheses", audit.auditId, pages.cursor];
  const query = useQuery({
    queryKey,
    enabled,
    queryFn: async ({ signal }) => {
      const data = requireData(
        await api.request((client) =>
          client.GET("/v1/audits/{auditId}/hypotheses", {
            signal,
            params: {
              path: { auditId: audit.auditId },
              query: {
                limit: 50,
                ...(pages.cursor ? { cursor: pages.cursor } : {}),
              },
            },
          }),
        ),
      );
      if (
        data.page.hasMore &&
        (data.items.length === 0 ||
          !data.page.nextCursor ||
          data.page.nextCursor === pages.cursor)
      )
        throw invalidAPIResponse(200, "Invalid hypothesis continuation");
      return data;
    },
    refetchInterval: pollInterval<components["schemas"]["AuditHypothesisPage"]>(
      {
        interval: 5000,
        settled: () => auditPollInterval([audit]) === false,
      },
    ),
  });
  useAuditProjectionRefresh(audit, queryKey, enabled);
  return { query, pages };
}

export function useResearchMaterials(audit: Audit, enabled = true) {
  const api = usePublicAPI();
  const pages = useURLCursorStack({ param: "materialsCursor" });
  const queryKey = [
    ...queryKeys.audits.detail(audit.auditId),
    "materials",
    pages.cursor,
  ];
  const query = useQuery({
    queryKey,
    enabled,
    queryFn: async ({ signal }) => {
      const data = requireData(
        await api.request((client) =>
          client.GET("/v1/audits/{auditId}/materials", {
            signal,
            params: {
              path: { auditId: audit.auditId },
              query: {
                limit: 50,
                ...(pages.cursor ? { cursor: pages.cursor } : {}),
              },
            },
          }),
        ),
      );
      if (
        data.page.hasMore &&
        (data.items.length === 0 ||
          !data.page.nextCursor ||
          data.page.nextCursor === pages.cursor)
      )
        throw invalidAPIResponse(200, "Invalid material continuation");
      return data;
    },
    refetchInterval: pollInterval<components["schemas"]["AuditMaterialPage"]>({
      interval: 5000,
      settled: () => auditPollInterval([audit]) === false,
    }),
  });
  useAuditProjectionRefresh(audit, queryKey, enabled);
  return { query, pages };
}

export function taskStatus(task: ResearchTask): {
  label: string;
  tone: StatusTone;
} {
  return task.gap_code
    ? { label: "Technical gap", tone: "warning" }
    : task.state === "settled"
      ? { label: "Collected", tone: "info" }
      : task.state === "selected"
        ? { label: "Run selected", tone: "progress" }
        : { label: "Accepted", tone: "idle" };
}
