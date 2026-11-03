import {
  useInfiniteQuery,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router";
import {
  getProjectFindingSource,
  listProjectFindingSourceHistory,
  listProjectFindingSources,
  type ProjectFinding,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { RecordedTime } from "../../../app/recorded-time";
import { Button } from "../../../ui";
import { artifactDetailPath } from "../../artifacts/paths";
import { SourceEvidence } from "../../issues/evidence-panel";
import "../../issues/issues.css";
import { ExactArtifactLink } from "./shared";

function SourceHistory({
  projectId,
  receiptId,
}: {
  projectId: string;
  receiptId: string;
}) {
  const api = usePublicAPI();
  const client = useQueryClient();
  const key = ["project-findings", projectId, "sources", receiptId, "history"];
  const refresh = () => client.resetQueries({ queryKey: key, exact: true });
  const records = useInfiniteQuery({
    queryKey: key,
    staleTime: Infinity,
    refetchOnWindowFocus: false,
    initialPageParam: undefined as string | undefined,
    queryFn: ({ pageParam }) =>
      listProjectFindingSourceHistory(
        api,
        projectId,
        receiptId,
        pageParam === undefined ? {} : { cursor: pageParam },
      ),
    getNextPageParam: (last) =>
      last.page.hasMore ? last.page.nextCursor : undefined,
    retry: false,
  });
  return (
    <section>
      <h4>Source history and check results</h4>
      <Button
        size="xs"
        disabled={records.isFetching}
        onClick={() => void refresh()}
      >
        Refresh history
      </Button>
      {records.error === null ? null : (
        <p role="alert">
          History could not be loaded: {records.error.message}{" "}
          <Button size="xs" onClick={() => void refresh()}>
            Retry
          </Button>
        </p>
      )}
      {records.isPending ? <p role="status">Loading history…</p> : null}
      <ul>
        {records.data?.pages
          .flatMap((page) => page.items)
          .map((record) => (
            <li
              key={`${record.artifact.namespace}/${record.artifact.name}@${record.artifact.revision}`}
            >
              <Link
                to={artifactDetailPath(
                  { kind: "project", id: projectId },
                  record.artifact,
                )}
              >
                Open immutable record
              </Link>{" "}
              · <RecordedTime value={record.createdAt} />
            </li>
          ))}
      </ul>
      {records.hasNextPage ? (
        <Button
          size="xs"
          disabled={records.isFetchingNextPage}
          onClick={() => void records.fetchNextPage()}
        >
          More history
        </Button>
      ) : null}
    </section>
  );
}
function SourceDetail({
  projectId,
  receiptId,
}: {
  projectId: string;
  receiptId: string;
}) {
  const api = usePublicAPI();
  const source = useQuery({
    queryKey: ["project-findings", projectId, "sources", receiptId],
    queryFn: () => getProjectFindingSource(api, projectId, receiptId),
    retry: false,
  });
  if (source.error !== null)
    return (
      <p role="alert">
        Source could not be loaded: {source.error.message}{" "}
        <Button size="xs" onClick={() => void source.refetch()}>
          Retry
        </Button>
      </p>
    );
  if (source.data === undefined) return <p role="status">Loading source…</p>;
  const value = source.data;
  return (
    <div>
      <p>{value.document.description}</p>
      <p className="ui-caption">
        Run {value.origin.runId}
        {value.origin.runDeleted ? " · Run deleted" : ""} ·{" "}
        {value.origin.workflow.name}@{value.origin.workflow.version}
      </p>
      {value.origin.audit === undefined ? null : (
        <p className="ui-caption">
          Original check: {value.origin.audit.auditId}
        </p>
      )}
      {value.projectProposal === undefined ? (
        <p role="alert">The Project source material is unavailable.</p>
      ) : (
        <ExactArtifactLink
          projectId={projectId}
          artifact={value.projectProposal}
          projectReadable
          label="Original claim"
        />
      )}
      <SourceEvidence source={value} />
      <SourceHistory projectId={projectId} receiptId={receiptId} />
    </div>
  );
}
export function FindingSources({ finding }: { finding: ProjectFinding }) {
  const api = usePublicAPI();
  const [selected, setSelected] = useState(finding.firstReceiptId);
  const client = useQueryClient();
  const key = [
    "project-findings",
    finding.projectId,
    finding.findingId,
    "sources",
  ];
  const refresh = () => client.resetQueries({ queryKey: key, exact: true });
  const sources = useInfiniteQuery({
    queryKey: key,
    staleTime: Infinity,
    refetchOnWindowFocus: false,
    initialPageParam: undefined as string | undefined,
    queryFn: ({ pageParam }) =>
      listProjectFindingSources(
        api,
        finding.projectId,
        finding.findingId,
        pageParam === undefined ? {} : { cursor: pageParam },
      ),
    getNextPageParam: (last) =>
      last.page.hasMore ? last.page.nextCursor : undefined,
    retry: false,
  });
  return (
    <section aria-label="Independent sources">
      <h3>Independent sources</h3>
      <Button
        size="xs"
        disabled={sources.isFetching}
        onClick={() => void refresh()}
      >
        Refresh sources
      </Button>
      <p className="ui-caption">
        Sources and check results keep their own evidence. They do not change
        the current analyst assessment.
      </p>
      {sources.error === null ? null : (
        <p role="alert">
          Sources could not be loaded: {sources.error.message}{" "}
          <Button size="xs" onClick={() => void refresh()}>
            Retry
          </Button>
        </p>
      )}
      <ul>
        {sources.data?.pages
          .flatMap((page) => page.items)
          .map((source) => (
            <li key={source.receiptId}>
              <Button
                size="xs"
                onClick={() => setSelected(source.receiptId)}
                aria-pressed={source.receiptId === selected}
              >
                {source.origin.workflow.name} ·{" "}
                <RecordedTime value={source.createdAt} />
              </Button>{" "}
              · {source.evidenceCount} evidence files
            </li>
          ))}
      </ul>
      {sources.hasNextPage ? (
        <Button
          size="xs"
          disabled={sources.isFetchingNextPage}
          onClick={() => void sources.fetchNextPage()}
        >
          More sources
        </Button>
      ) : null}
      <SourceDetail
        key={selected}
        projectId={finding.projectId}
        receiptId={selected}
      />
    </section>
  );
}
