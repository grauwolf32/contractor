import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router";
import { MAXIMUM_PREVIEW_BYTES } from "../../../api/artifacts";
import { readAuditArtifact } from "../../../api/audit-artifacts";
import type { Audit } from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { queryKeys } from "../../../api/query-keys";
import { saveBlob } from "../../../app/download";
import { ErrorNotice } from "../../../app/error-notice";
import { Icon } from "../../../app/icon";
import { QueryView } from "../../../app/query-view";
import {
  Button,
  Facts,
  IconButton,
  StatusChip,
  TechnicalDetails,
} from "../../../ui";
import { LoadedArtifactPreview } from "../../artifacts/loaded-preview";
import type { ResearchTask } from "./research-data";
import { checkPath } from "./check-links";
import { issuePath } from "../../issues/links";
import { scopedTaskResult, taskResultPreview } from "./task-result";

function ReadResult({ audit, task }: { audit: Audit; task: ResearchTask }) {
  const api = usePublicAPI();
  const result = task.result!;
  const read = useQuery({
    queryKey: [
      ...queryKeys.audits.detail(audit.auditId),
      "task-result",
      task.document.task_id,
      result.ref.revision,
    ],
    queryFn: async ({ signal }) => {
      const value = await readAuditArtifact(
        api,
        audit.auditId,
        `task/${task.document.task_id}/result`,
        result.ref.revision,
        undefined,
        signal,
      );
      const text = await value.blob.text();
      return {
        ...value,
        text,
        scoped:
          value.mediaType === "application/json"
            ? scopedTaskResult(text, task)
            : undefined,
      };
    },
    staleTime: Infinity,
  });
  return (
    <QueryView
      query={read}
      loading={<p role="status">Loading saved result…</p>}
    >
      {(data) => {
        const item = data.scoped;
        return item ? (
          <section className="research-result" aria-label="Scoped check result">
            <header>
              <h3>Recorded result</h3>
              {item.verdict ? (
                <StatusChip
                  tone={item.verdict === "refuted" ? "success" : "warning"}
                >
                  {item.verdict === "supported"
                    ? "Supported"
                    : item.verdict === "refuted"
                      ? "Refuted"
                      : "Inconclusive"}
                </StatusChip>
              ) : (
                <StatusChip tone="neutral">Observations</StatusChip>
              )}
            </header>
            <p>{item.rationale}</p>
            <Facts
              layout="list"
              items={[
                {
                  term: "Tested conditions",
                  value: item.tested_conditions.length ? (
                    <ul>
                      {item.tested_conditions.map((s) => (
                        <li key={s}>{s}</li>
                      ))}
                    </ul>
                  ) : (
                    "No conditions recorded"
                  ),
                },
                {
                  term: "Limitations",
                  value: item.limitations.length ? (
                    <ul>
                      {item.limitations.map((s) => (
                        <li key={s}>{s}</li>
                      ))}
                    </ul>
                  ) : (
                    "No limitations recorded"
                  ),
                },
              ]}
            />
            <p className="ui-caption">
              {item.evidence.length} exact evidence references. Results apply to
              the recorded conditions; a negative recheck does not change an
              issue assessment.
            </p>
            <Link to={`${checkPath(audit.projectId, audit.auditId)}/materials`}>
              Inspect retained evidence and materials →
            </Link>
            <TechnicalDetails description="Full per-task result and trusted origin.">
              <pre className="checks-task-json">
                {taskResultPreview(data.text, task.document.task_id)}
              </pre>
            </TechnicalDetails>
          </section>
        ) : (
          <LoadedArtifactPreview
            mediaType={data.mediaType}
            source={taskResultPreview(data.text, task.document.task_id)}
          />
        );
      }}
    </QueryView>
  );
}

export function TaskResultView({
  audit,
  task,
  initiallyOpen = false,
}: {
  audit: Audit;
  task: ResearchTask;
  initiallyOpen?: boolean;
}) {
  const api = usePublicAPI();
  const [open, setOpen] = useState(initiallyOpen);
  const result = task.result;
  const download = useMutation({
    mutationFn: async () => {
      if (!result) throw new Error("No collected result");
      return readAuditArtifact(
        api,
        audit.auditId,
        `task/${task.document.task_id}/result`,
        result.ref.revision,
        undefined,
        new AbortController().signal,
      );
    },
    onSuccess: (data) => saveBlob(data.blob, `${task.document.task_id}.result`),
  });
  if (task.findings)
    return (
      <section
        className="research-finding-results"
        aria-label="Finding results"
      >
        {task.findings.ids.length ? (
          <ul>
            {task.findings.ids.map((id) => (
              <li key={id}>
                <Link to={issuePath(audit.projectId, id)}>{id}</Link>
              </li>
            ))}
          </ul>
        ) : (
          <p className="ui-caption">No Findings recorded for this check.</p>
        )}
        {task.findings.truncated ? (
          <p className="ui-caption">
            More Findings are available in the check’s Findings list.
          </p>
        ) : null}
      </section>
    );
  if (!result) return <p className="ui-caption">No collected result yet.</p>;
  return (
    <div className="research-saved-result">
      <div className="ui-toolbar">
        <Button
          size="sm"
          icon={<Icon name="open" />}
          onClick={() => setOpen(!open)}
        >
          {open ? "Hide saved result" : "Read saved result"}
        </Button>
        <IconButton
          size="sm"
          label="Download complete Run result"
          icon={<Icon name="download" />}
          pending={download.isPending}
          onClick={() => download.mutate()}
        />
      </div>
      {download.error ? <ErrorNotice error={download.error} /> : null}
      {open ? (
        result.size_bytes > MAXIMUM_PREVIEW_BYTES ? (
          <p className="ui-caption">
            The saved result exceeds the inline preview limit. Download its
            exact bytes.
          </p>
        ) : (
          <ReadResult audit={audit} task={task} />
        )
      ) : null}
    </div>
  );
}
