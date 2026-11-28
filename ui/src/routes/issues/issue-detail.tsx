import { pollInterval } from "../../app/polling";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useId, useLayoutEffect, useRef } from "react";
import { Link, useSearchParams } from "react-router";
import {
  getProjectFinding,
  findingSeverity,
  type ProjectFindingEditResult,
} from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import type { CrossProjectIssue } from "../../api/cross-project";
import { getProject, type Project } from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import { ContextLink } from "../../app/context-navigation";
import { Icon } from "../../app/icon";
import { QueryView } from "../../app/query-view";
import { capitalize, TERMS } from "../../app/vocabulary";
import { Button, DetailPane, TechnicalDetails } from "../../ui";
import { FindingDecision } from "../decisions";
import { FindingSources } from "../projects/audits/finding-card";
import { EvidencePanel } from "./evidence-panel";
import { useFindingSource } from "./source-data";
import { evidenceCount } from "./evidence";
import {
  IssueIdentity,
  IssueOverview,
  IssueRecordedDecision,
} from "./overview";
import { TabPanel, Tabs } from "./tabs";
import { FactOverview } from "./fact-overview";
import "./issues.css";

type IssueTab = "summary" | "evidence" | "history";
interface IssueDetailProps {
  projectId: string;
  findingId: string;
  listed?: CrossProjectIssue | undefined;
  projects: readonly Project[];
  position?: { index: number; count: number } | undefined;
  listPending?: boolean | undefined;
  onPrevious?: (() => void) | undefined;
  onNext?: (() => void) | undefined;
  onDecided?: ((result: ProjectFindingEditResult) => void) | undefined;
  factsReturn?: string;
  focusRequest?: number | undefined;
}
export function IssueDetail({
  projectId,
  findingId,
  listed,
  projects,
  position,
  listPending = false,
  onPrevious,
  onNext,
  onDecided,
  factsReturn,
  focusRequest,
}: IssueDetailProps) {
  const api = usePublicAPI();
  const client = useQueryClient();
  const idBase = useId();
  const header = useRef<HTMLElement>(null);
  const [params, setParams] = useSearchParams();
  const selectedTab = params.get("tab");
  const tab: IssueTab =
    selectedTab === "evidence" || selectedTab === "history"
      ? selectedTab
      : "summary";
  const setTab = (value: IssueTab) => {
    const next = new URLSearchParams(params);
    if (value === "summary") next.delete("tab");
    else next.set("tab", value);
    void setParams(next, { replace: true });
  };
  const finding = useQuery({
    queryKey: ["project-findings", projectId, findingId],
    queryFn: () => getProjectFinding(api, projectId, findingId),
    refetchInterval: pollInterval({ interval: 20_000 }),
    retry: false,
  });
  const known =
    projects.find((candidate) => candidate.projectId === projectId) ??
    listed?.project;
  const project = useQuery({
    queryKey: queryKeys.projects.detail(projectId),
    queryFn: () => getProject(api, projectId),
    enabled: known === undefined,
    retry: false,
  });
  const source = useFindingSource(finding.data);
  useLayoutEffect(() => {
    if (focusRequest !== undefined) header.current?.focus();
  }, [focusRequest]);
  const listedRevision = listed?.finding.revision;
  const shownRevision = finding.data?.revision;
  useEffect(() => {
    if (
      listedRevision !== undefined &&
      shownRevision !== undefined &&
      listedRevision > shownRevision
    )
      void client.invalidateQueries({
        queryKey: ["project-findings", projectId, findingId],
        exact: true,
      });
  }, [projectId, findingId, listedRevision, shownRevision, client]);
  const shown = finding.data;
  const identity = shown ?? listed?.finding;
  const fact =
    identity !== undefined && findingSeverity(identity) === "informational";
  const subject =
    identity &&
    ("subjectKey" in identity
      ? identity.subjectKey
      : identity.claim?.subject?.key);
  const kind = fact
    ? "Fact"
    : shown?.state === "confirmed"
      ? capitalize(TERMS.issue)
      : capitalize(TERMS.possibleIssue);
  const bar = (
    <header ref={header} tabIndex={-1} className="issues-detail-bar">
      <div className="issues-detail-top">
        <p className="issues-detail-context">
          {known?.name ?? project.data?.name ?? projectId}
        </p>
        <nav
          className="issues-detail-nav"
          aria-label={factsReturn ? "Fact navigation" : "Issue navigation"}
        >
          <Button
            className="issues-step"
            size="xs"
            aria-label={fact ? "Previous fact" : "Previous possible issue"}
            aria-keyshortcuts="K"
            disabled={onPrevious === undefined}
            onClick={onPrevious}
          >
            ‹ Prev
          </Button>
          <span
            className="issues-detail-position"
            aria-label={
              position === undefined
                ? kind
                : `${kind} ${position.index + 1} of ${position.count}`
            }
          >
            <span className="ui-visually-hidden">{kind} </span>
            {position === undefined
              ? shown === undefined
                ? "Loading…"
                : listPending
                  ? "Loading the list…"
                  : "Not in this list"
              : `${position.index + 1} of ${position.count}`}
          </span>
          <Button
            className="issues-step"
            size="xs"
            aria-label={fact ? "Next fact" : "Next possible issue"}
            aria-keyshortcuts="J"
            disabled={onNext === undefined}
            onClick={onNext}
          >
            Next ›
          </Button>
          {factsReturn ? (
            <Link className="fact-back" to={factsReturn}>
              Back to facts
            </Link>
          ) : (
            <ContextLink
              returnLabel={fact ? "Fact" : "Possible issue"}
              className="issues-detail-check"
              to={`/projects/${encodeURIComponent(projectId)}/${fact ? "facts" : "findings"}`}
            >
              View in Project
              <Icon name="chevron-right" size={14} />
            </ContextLink>
          )}
        </nav>
      </div>
      {identity === undefined ? null : (
        <>
          {fact ? (
            <>
              <h2 className="issues-workspace-title">{identity.title}</h2>
              <div className="fact-context">
                <span>Informational observation</span>
                {subject && <code>{subject}</code>}
              </div>
            </>
          ) : (
            <IssueIdentity finding={identity} />
          )}
          <Tabs
            label="Review sections"
            items={[
              { id: "summary", label: "Summary" },
              {
                id: "evidence",
                label: "Evidence",
                ...(source.data === undefined
                  ? {}
                  : { count: evidenceCount(source.data) }),
              },
              { id: "history", label: "History" },
            ]}
            selected={tab}
            onSelect={setTab}
            idBase={idBase}
          />
        </>
      )}
    </header>
  );
  return (
    <DetailPane
      header={bar}
      footer={
        shown === undefined || fact ? undefined : (
          <FindingDecision
            finding={shown}
            presentation="review"
            onDecided={onDecided}
          />
        )
      }
    >
      <QueryView
        query={finding}
        loading={
          <p className="ui-caption" role="status">
            {factsReturn ? "Loading fact…" : "Loading the possible issue…"}
          </p>
        }
        errorContext={
          factsReturn
            ? "This observation could not be loaded."
            : "This possible issue could not be loaded."
        }
      >
        {(value) => (
          <>
            {!fact && <IssueRecordedDecision finding={value} />}
            <TabPanel idBase={idBase} id={tab}>
              {tab === "summary" && fact ? (
                <FactOverview
                  key={value.findingId}
                  finding={value}
                  onDecided={onDecided}
                />
              ) : tab === "summary" ? (
                <IssueOverview
                  finding={value}
                  supporting={
                    <TechnicalDetails>
                      <p>
                        Project {projectId} · Finding {findingId} · Revision{" "}
                        {value.revision}
                      </p>
                    </TechnicalDetails>
                  }
                />
              ) : tab === "evidence" ? (
                <EvidencePanel finding={value} />
              ) : (
                <FindingSources finding={value} />
              )}
            </TabPanel>
          </>
        )}
      </QueryView>
    </DetailPane>
  );
}
