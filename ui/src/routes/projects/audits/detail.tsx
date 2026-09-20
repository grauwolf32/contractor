import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState, type ReactNode } from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router";

import {
  AUDIT_ID_PATTERN,
  auditNeedsPolling,
  getAudit,
  type Audit,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { PublicAPIError } from "../../../api/error";
import { getProject, PROJECT_ID_PATTERN } from "../../../api/projects";
import { queryKeys } from "../../../api/query-keys";
import { ReturnLink } from "../../../app/context-navigation";
import { MobileSectionPicker } from "../../../app/mobile-section-picker";
import { ErrorNotice, formatTimestamp } from "../../artifacts/common";
import { StateBadge } from "../../runs/components";
import { AuditFindings } from "./audit-findings";
import { AuditControls } from "./controls";
import { AuditCoverage } from "./coverage";
import { AuditChecks, AuditRuns } from "./executions";
import { auditProfileLabel } from "./labels";
import { AuditOverview } from "./overview";
import { AuditReportView } from "./report";
import { AuditReviews } from "./reviews";

export { AuditFindingCard } from "./finding-card";

import "./styles.css";

type AuditSection =
  | "overview"
  | "coverage"
  | "findings"
  | "checks"
  | "reviews"
  | "runs"
  | "report";

const SECTIONS: readonly { id: AuditSection; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "coverage", label: "Coverage" },
  { id: "findings", label: "Findings" },
  { id: "checks", label: "Checks" },
  { id: "reviews", label: "Reviews" },
  { id: "runs", label: "Runs" },
  { id: "report", label: "Report" },
];

function AuditSectionContent({
  audit,
  section,
  api,
}: {
  audit: Audit;
  section: AuditSection;
  api: ReturnType<typeof usePublicAPI>;
}) {
  switch (section) {
    case "overview":
      return <AuditOverview audit={audit} />;
    case "coverage":
      return <AuditCoverage audit={audit} />;
    case "findings":
      return <AuditFindings audit={audit} />;
    case "checks":
      return <AuditChecks audit={audit} api={api} />;
    case "reviews":
      return <AuditReviews audit={audit} />;
    case "runs":
      return <AuditRuns audit={audit} api={api} />;
    case "report":
      return <AuditReportView audit={audit} api={api} />;
  }
}

function AuditIdentity({ audit }: { audit: Audit }) {
  const [copyStatus, setCopyStatus] = useState("");
  return (
    <div className="audit-identity">
      <time dateTime={audit.createdAt}>{formatTimestamp(audit.createdAt)}</time>
      <code>{audit.auditId}</code>
      <button
        type="button"
        className="secondary-button"
        onClick={() => {
          void navigator.clipboard.writeText(audit.auditId).then(
            () => setCopyStatus("Copied Audit ID"),
            () =>
              setCopyStatus(
                "Copy unavailable. Select the Audit ID to copy it manually.",
              ),
          );
        }}
      >
        Copy Audit ID
      </button>
      <span role="status">{copyStatus}</span>
    </div>
  );
}

export function ProjectAuditDetailRoute() {
  const location = useLocation();
  const api = usePublicAPI();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { projectId = "", auditId = "", section: rawSection } = useParams();
  const validRoute =
    PROJECT_ID_PATTERN.test(projectId) && AUDIT_ID_PATTERN.test(auditId);
  const section = SECTIONS.some((candidate) => candidate.id === rawSection)
    ? (rawSection as AuditSection)
    : "overview";
  const project = useQuery({
    queryKey: queryKeys.projects.detail(projectId),
    queryFn: () => getProject(api, projectId),
    enabled: validRoute,
  });
  const audit = useQuery({
    queryKey: queryKeys.audits.detail(auditId),
    queryFn: () => getAudit(api, auditId),
    enabled: validRoute,
    refetchInterval: (query) =>
      query.state.data !== undefined &&
      auditNeedsPolling(query.state.data.state)
        ? 1_000
        : false,
    refetchOnReconnect: true,
    refetchOnWindowFocus: true,
    retry: (count, error) =>
      !(error instanceof PublicAPIError && error.status === 404) && count < 2,
  });
  useEffect(() => {
    if (audit.error instanceof PublicAPIError && audit.error.status === 404) {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.projects.audits.all(projectId),
      });
      void navigate(`/projects/${encodeURIComponent(projectId)}/audits`, {
        replace: true,
      });
    }
  }, [audit.error, navigate, projectId, queryClient]);

  let content: ReactNode;
  if (!validRoute)
    content = (
      <ErrorNotice error={new Error("Project Audit route is invalid")} />
    );
  else if (audit.isPending)
    content = <p className="loading-copy">Loading Audit…</p>;
  else if (audit.error !== null) content = <ErrorNotice error={audit.error} />;
  else if (audit.data.projectId !== projectId)
    content = (
      <ErrorNotice error={new Error("Audit does not belong to this Project")} />
    );
  else
    content = (
      <AuditSectionContent audit={audit.data} section={section} api={api} />
    );

  return (
    <section className="route-page audit-page audit-detail-page">
      <header className="route-header-row">
        <div>
          <ReturnLink
            to={`/projects/${encodeURIComponent(projectId)}/audits`}
            label={`${project.data?.name ?? "Project"} Audits`}
          />
          <p className="eyebrow">{project.data?.name ?? "Audit"}</p>
          <h2>
            {audit.data === undefined ? "Audit" : auditProfileLabel(audit.data)}
          </h2>
          {audit.data === undefined ? (
            <code>{auditId}</code>
          ) : (
            <AuditIdentity audit={audit.data} />
          )}
        </div>
        {audit.data === undefined ? null : (
          <div className="audit-header-state">
            <StateBadge state={audit.data.state} />
            <span>revision {audit.data.revision}</span>
          </div>
        )}
      </header>
      {project.error === null ? null : <ErrorNotice error={project.error} />}
      {audit.data === undefined ? null : (
        <AuditControls audit={audit.data} projectName={project.data?.name} />
      )}
      <nav
        className="audit-section-navigation section-navigation"
        aria-label="Audit sections"
      >
        {SECTIONS.map((candidate) => {
          const target =
            candidate.id === "overview"
              ? `/projects/${encodeURIComponent(projectId)}/audits/${encodeURIComponent(auditId)}`
              : `/projects/${encodeURIComponent(projectId)}/audits/${encodeURIComponent(auditId)}/${candidate.id}`;
          return (
            <Link
              key={candidate.id}
              className={section === candidate.id ? "active" : ""}
              aria-current={section === candidate.id ? "page" : undefined}
              to={target}
              state={location.state}
            >
              {candidate.label}
            </Link>
          );
        })}
      </nav>
      <MobileSectionPicker
        label="Audit section"
        value={
          section === "overview"
            ? `/projects/${encodeURIComponent(projectId)}/audits/${encodeURIComponent(auditId)}`
            : `/projects/${encodeURIComponent(projectId)}/audits/${encodeURIComponent(auditId)}/${section}`
        }
        options={SECTIONS.map((candidate) => ({
          to: `/projects/${encodeURIComponent(projectId)}/audits/${encodeURIComponent(auditId)}${candidate.id === "overview" ? "" : `/${candidate.id}`}`,
          label: candidate.label,
        }))}
        state={location.state}
      />
      {content}
    </section>
  );
}
