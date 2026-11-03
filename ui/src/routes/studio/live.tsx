import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router";
import {
  AUDIT_ID_PATTERN,
  auditPollInterval,
  getAudit,
  type Audit,
} from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import { pollInterval } from "../../app/polling";
import { QueryView } from "../../app/query-view";
import { Button, Facts, Field } from "../../ui";
import type { Draft } from "./document";

/** The live view reads the common owner state; tasks/results open in the check. */
export function StudioLive({
  initialAudit,
}: {
  initialAudit: string;
  draft: Draft;
}) {
  const api = usePublicAPI();
  const [input, setInput] = useState(initialAudit);
  const [auditId, setAuditId] = useState(
    AUDIT_ID_PATTERN.test(initialAudit) ? initialAudit : "",
  );
  const query = useQuery({
    queryKey: queryKeys.audits.detail(auditId),
    enabled: auditId !== "",
    queryFn: ({ signal }) => getAudit(api, auditId, signal),
    refetchInterval: pollInterval<Audit>({
      interval: 1000,
      settled: (audit) => auditPollInterval([audit]) === false,
    }),
  });
  return (
    <div className="ui-stack">
      <Field label="Check ID">
        {(control) => (
          <input
            {...control}
            value={input}
            onChange={(event) => setInput(event.target.value)}
          />
        )}
      </Field>
      <Button
        disabled={!AUDIT_ID_PATTERN.test(input)}
        onClick={() => setAuditId(input)}
      >
        Load check
      </Button>
      {auditId === "" ? (
        <p>Select a check to read its current state.</p>
      ) : (
        <QueryView query={query} loading={<p role="status">Loading check…</p>}>
          {(audit) => (
            <>
              <Facts
                items={[
                  { term: "State", value: audit.state },
                  { term: "Accepted tasks", value: audit.taskCount },
                  {
                    term: "Outstanding Runs",
                    value: audit.outstandingRunCount,
                  },
                ]}
              />
              <Link
                to={`/projects/${encodeURIComponent(audit.projectId)}/audits/${encodeURIComponent(audit.auditId)}`}
              >
                Open tasks and saved results
              </Link>
            </>
          )}
        </QueryView>
      )}
    </div>
  );
}
