import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import { useNavigate, useParams } from "react-router";
import {
  AUDIT_ID_PATTERN,
  auditPollInterval,
  getAudit,
  type Audit,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { invalidateCrossProject } from "../../../api/cross-project";
import { PublicAPIError } from "../../../api/error";
import { PROJECT_ID_PATTERN } from "../../../api/projects";
import { queryKeys } from "../../../api/query-keys";
import { ErrorNotice } from "../../../app/error-notice";
import { pollInterval } from "../../../app/polling";
import { QueryView } from "../../../app/query-view";
import { ResearchAudit } from "./research";

/** Common research journey over authoritative Tasks, hypotheses and exact materials. */
export function ProjectAuditDetailRoute() {
  const { projectId = "", auditId = "" } = useParams();
  const api = usePublicAPI();
  const valid =
    PROJECT_ID_PATTERN.test(projectId) && AUDIT_ID_PATTERN.test(auditId);
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const audit = useQuery({
    queryKey: queryKeys.audits.detail(auditId),
    enabled: valid,
    queryFn: ({ signal }) => getAudit(api, auditId, signal),
    refetchInterval: pollInterval<Audit>({
      interval: 1000,
      settled: (current) => auditPollInterval([current]) === false,
    }),
  });
  useEffect(() => {
    if (audit.error instanceof PublicAPIError && audit.error.status === 404) {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.projects.audits.all(projectId),
      });
      void invalidateCrossProject(queryClient);
      void navigate(`/projects/${encodeURIComponent(projectId)}/audits`, {
        replace: true,
      });
    }
  }, [audit.error, queryClient, navigate, projectId]);
  if (!valid)
    return <ErrorNotice error={new Error("This check address is invalid.")} />;
  return (
    <QueryView query={audit} loading={<p role="status">Loading check…</p>}>
      {(value) =>
        value.projectId === projectId ? (
          <ResearchAudit key={value.auditId} audit={value} />
        ) : (
          <ErrorNotice
            error={new Error("This check does not belong to this project.")}
          />
        )
      }
    </QueryView>
  );
}
