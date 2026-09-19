import { useQueries } from "@tanstack/react-query";
import { Link } from "react-router";
import { type Audit, getAuditWorkspace } from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";

export function ProjectFindingSummary({
  audits,
  projectId,
}: {
  audits: Audit[];
  projectId: string;
}) {
  const api = usePublicAPI();
  const summaries = useQueries({
    queries: audits.map((audit) => ({
      queryKey: [...queryKeys.audits.detail(audit.auditId), "workspace"],
      queryFn: () => getAuditWorkspace(api, audit.auditId),
      retry: false,
      staleTime: 10000,
    })),
  });
  const count = summaries.reduce(
    (total, query) => total + (query.data?.unreviewedFindings ?? 0),
    0,
  );
  const complete = summaries.every((query) => query.isSuccess);
  return (
    <div
      className={`project-finding-summary ${count > 0 ? "has-findings" : ""}`}
    >
      {summaries.some((query) => query.isPending) ? (
        <p>Checking findings in recent audits…</p>
      ) : (
        <p>
          {complete
            ? `${count} unreviewed findings in ${audits.length} recent audits.`
            : "Finding counts are unavailable for some recent audits."}
        </p>
      )}
      <Link
        to={`/projects/${encodeURIComponent(projectId)}/findings?verdict=unreviewed`}
      >
        Review project findings →
      </Link>
    </div>
  );
}
