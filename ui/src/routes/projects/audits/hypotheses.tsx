import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { Link, useSearchParams } from "react-router";
import { auditPollInterval, type Audit } from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { PublicAPIError, requireData } from "../../../api/error";
import type { components } from "../../../api/generated/public";
import { queryKeys } from "../../../api/query-keys";
import { CursorControls } from "../../../app/cursor-controls";
import { ErrorNotice } from "../../../app/error-notice";
import { useURLCursorStack } from "../../../app/pagination";
import { pollInterval } from "../../../app/polling";
import { QueryView } from "../../../app/query-view";
import { createMutationIdempotencyKey } from "../../../mutations/idempotency";
import { Button, Field, TechnicalDetails } from "../../../ui";

type Hypothesis = components["schemas"]["AuditHypothesis"];
type Content = components["schemas"]["AuditHypothesisContent"];
type Page = components["schemas"]["AuditHypothesisPage"];
type PendingWrite = {
  method?: string;
  key: string;
  revision: number;
  content: Content;
  id?: string;
};

/** Assertions are owner commands over ordinary retained Task results. */
export function AuditHypotheses({ audit }: { audit: Audit }) {
  const api = usePublicAPI();
  const [search] = useSearchParams();
  const cache = useQueryClient();
  const [open, setOpen] = useState(false);
  const [editing, setEditing] = useState<Hypothesis>();
  const [statement, setStatement] = useState("");
  const [conditions, setConditions] = useState("");
  const [grounds, setGrounds] = useState("");
  const [busy, setBusy] = useState(false);
  const [writeLocked, setWriteLocked] = useState(false);
  const [checkLocked, setCheckLocked] = useState(false);
  const [error, setError] = useState<unknown>();
  const [conflict, setConflict] = useState(false);
  const [message, setMessage] = useState("");
  const pending = useRef<PendingWrite | undefined>(undefined);
  const pages = useURLCursorStack({ param: "hypothesesCursor" });
  const editable = audit.state === "active" || audit.state === "paused";
  const query = useQuery({
    queryKey: ["audit-hypotheses", audit.auditId, audit.revision, pages.cursor],
    enabled: open,
    queryFn: async ({ signal }) =>
      requireData(
        await api.request((client) =>
          client.GET("/v1/audits/{auditId}/hypotheses", {
            signal,
            params: {
              path: { auditId: audit.auditId },
              query: {
                limit: 50,
                ...(pages.cursor === undefined ? {} : { cursor: pages.cursor }),
              },
            },
          }),
        ),
      ),
    refetchInterval: pollInterval<Page>({
      interval: 2000,
      settled: () => auditPollInterval([audit]) === false,
    }),
  });
  const methods = query.data?.checking_workflows ?? [];

  async function refresh() {
    await Promise.all([
      cache.invalidateQueries({
        queryKey: queryKeys.audits.detail(audit.auditId),
      }),
      cache.invalidateQueries({
        queryKey: ["audit-hypotheses", audit.auditId],
      }),
    ]);
  }

  function choose(value?: Hypothesis) {
    if (pending.current !== undefined || busy) return;
    setEditing(value);
    setStatement(value?.document.statement ?? "");
    setConditions(value?.document.scope.conditions ?? "");
    setGrounds(value?.document.grounds ?? "");
    setConflict(false);
    setError(undefined);
    setMessage("");
  }

  async function save(method?: string) {
    const request = pending.current ?? {
      key: createMutationIdempotencyKey("hypothesis"),
      ...(method === undefined ? {} : { method }),
      revision: audit.revision,
      content: {
        key: editing?.source_key ?? createMutationIdempotencyKey("assertion"),
        statement,
        grounds,
        scope: {
          ...Object.fromEntries(
            Object.entries(editing?.document.scope ?? {}).filter(
              ([key]) => key !== "conditions",
            ),
          ),
          ...(conditions.trim() === "" ? {} : { conditions }),
        },
      },
      ...(editing === undefined ? {} : { id: editing.document.id }),
    };
    pending.current = request;
    setWriteLocked(true);
    setBusy(true);
    setError(undefined);
    try {
      const params = {
        path: { auditId: audit.auditId },
        header: {
          "Idempotency-Key": request.key,
          "If-Match": `"${request.revision}"`,
        },
      };
      if (request.method !== undefined) {
        const taskKey = request.key;
        requireData(
          await api.request((client) =>
            client.POST("/v1/audits/{auditId}/tasks", {
              params,
              body: {
                work: [
                  {
                    key: taskKey,
                    workflow: request.method!,
                    objective:
                      "Check the selected assertion under its stated conditions",
                    inputs: {},
                  },
                ],
                hypotheses:
                  request.id === undefined ? {} : { [taskKey]: request.id },
                assertions: { [taskKey]: request.content },
                attention: [taskKey],
              },
            }),
          ),
        );
      } else if (request.id === undefined) {
        requireData(
          await api.request((client) =>
            client.POST("/v1/audits/{auditId}/hypotheses", {
              params,
              body: request.content,
            }),
          ),
        );
      } else {
        requireData(
          await api.request((client) =>
            client.PATCH("/v1/audits/{auditId}/hypotheses/{hypothesisId}", {
              params: {
                ...params,
                path: { ...params.path, hypothesisId: request.id! },
              },
              body: request.content,
            }),
          ),
        );
      }
      pending.current = undefined;
      setWriteLocked(false);
      setEditing(undefined);
      setStatement("");
      setConditions("");
      setGrounds("");
      setMessage(
        request.method === undefined
          ? "Assertion saved. Existing checks retain their original wording and conditions."
          : "Assertion saved and check accepted. Existing checks retain their original wording and conditions.",
      );
      await refresh();
    } catch (reason) {
      setError(reason);
      if (
        reason instanceof PublicAPIError &&
        reason.status >= 400 &&
        reason.status < 500 &&
        reason.status !== 408 &&
        reason.status !== 429
      ) {
        pending.current = undefined;
        setWriteLocked(false);
        setConflict(reason.status === 412);
        await refresh();
      }
    } finally {
      setBusy(false);
    }
  }

  const checking = useRef<
    { key: string; revision: number; id: string; method: string } | undefined
  >(undefined);
  async function check(id: string, method: string) {
    const request = checking.current ?? {
      key: createMutationIdempotencyKey("check-assertion"),
      revision: audit.revision,
      id,
      method,
    };
    checking.current = request;
    setCheckLocked(true);
    setBusy(true);
    setError(undefined);
    try {
      const taskKey = request.key;
      requireData(
        await api.request((client) =>
          client.POST("/v1/audits/{auditId}/tasks", {
            params: {
              path: { auditId: audit.auditId },
              header: {
                "Idempotency-Key": request.key,
                "If-Match": `"${request.revision}"`,
              },
            },
            body: {
              work: [
                {
                  key: taskKey,
                  workflow: request.method,
                  objective:
                    "Check the selected assertion under its stated conditions",
                  inputs: {},
                },
              ],
              hypotheses: { [taskKey]: request.id },
              attention: [taskKey],
            },
          }),
        ),
      );
      checking.current = undefined;
      setCheckLocked(false);
      setMessage(
        "Check accepted. Its progress and saved result appear in the task list.",
      );
      await refresh();
    } catch (reason) {
      setError(reason);
      if (
        reason instanceof PublicAPIError &&
        reason.status >= 400 &&
        reason.status < 500 &&
        reason.status !== 408 &&
        reason.status !== 429
      ) {
        checking.current = undefined;
        setCheckLocked(false);
        await refresh();
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="ui-stack" aria-label="Assertions">
      <Button size="sm" onClick={() => setOpen(!open)}>
        {open ? "Hide assertions" : "Show assertions"}
      </Button>
      {open ? (
        <>
          <QueryView
            query={query}
            loading={<p role="status">Loading assertions…</p>}
          >
            {(page) => (
              <>
                <p>
                  {page.fixed_denominator} requirements in the original
                  checklist. New assertions, split checks and missing evidence
                  do not change this total.
                </p>
                {page.pending_fixed_count === 0 ? null : (
                  <p role="status">
                    {page.pending_fixed_count} original requirements have no
                    accepted assertion yet. They remain unchecked.
                  </p>
                )}
                {page.items.map((value) => (
                  <article
                    className="checks-task ui-stack"
                    key={value.document.id}
                  >
                    <h2>{value.document.statement}</h2>
                    <p>{value.document.grounds}</p>
                    {value.document.scope.conditions === undefined ? null : (
                      <p>{value.document.scope.conditions}</p>
                    )}
                    {value.document.scope.coverage_gap === undefined ? null : (
                      <p>{value.document.scope.coverage_gap}</p>
                    )}
                    <Link
                      to={`?${new URLSearchParams([...search.entries()].filter(([key]) => key !== "tasksCursor" && key !== "hypothesis").concat([["hypothesis", value.document.id]]))}`}
                    >
                      View checks and results
                    </Link>
                    <p>
                      {value.fixed
                        ? "Original checklist requirement"
                        : "Additional assertion"}
                    </p>
                    {editable ? (
                      <div className="ui-toolbar">
                        <Button
                          size="sm"
                          disabled={busy || writeLocked || checkLocked}
                          onClick={() => choose(value)}
                        >
                          Edit assertion
                        </Button>
                        {methods.map((method) => (
                          <Button
                            key={method}
                            size="sm"
                            disabled={busy || writeLocked || checkLocked}
                            onClick={() =>
                              void check(value.document.id, method)
                            }
                          >
                            Check assertion
                            {methods.length > 1 ? ` · ${method}` : ""}
                          </Button>
                        ))}
                      </div>
                    ) : null}
                    <TechnicalDetails description="Exact assertion revision, source, scope and evidence requirements.">
                      <pre className="checks-task-json">
                        {JSON.stringify(value, null, 2)}
                      </pre>
                    </TechnicalDetails>
                  </article>
                ))}
                <CursorControls
                  label="Assertion pages"
                  {...pages.controls(page.page)}
                />
              </>
            )}
          </QueryView>
          {editable ? (
            <form
              className="ui-stack"
              onSubmit={(event) => {
                event.preventDefault();
                void save();
              }}
            >
              <h2>
                {editing === undefined ? "Add an assertion" : "Edit assertion"}
              </h2>
              <Field
                label="Assertion"
                required
                hint="One observable claim to check under stated conditions."
              >
                {(control) => (
                  <textarea
                    {...control}
                    value={statement}
                    disabled={busy || writeLocked}
                    onChange={(event) => setStatement(event.target.value)}
                  />
                )}
              </Field>
              <Field label="Conditions" optional>
                {(control) => (
                  <textarea
                    {...control}
                    value={conditions}
                    disabled={busy || writeLocked}
                    onChange={(event) => setConditions(event.target.value)}
                  />
                )}
              </Field>
              <Field label="Why check this?" required>
                {(control) => (
                  <textarea
                    {...control}
                    value={grounds}
                    disabled={busy || writeLocked}
                    onChange={(event) => setGrounds(event.target.value)}
                  />
                )}
              </Field>
              {conflict ? (
                <p role="alert">
                  {editing === undefined
                    ? "This check changed. Review your new assertion and use the refreshed check revision before retrying."
                    : "This check changed. Select Edit assertion again to load its current content before saving."}
                </p>
              ) : null}
              <div className="ui-toolbar">
                {conflict && editing === undefined ? (
                  <Button disabled={busy} onClick={() => setConflict(false)}>
                    Use refreshed revision
                  </Button>
                ) : null}
                <Button
                  type="submit"
                  disabled={
                    busy ||
                    conflict ||
                    checkLocked ||
                    statement.trim() === "" ||
                    grounds.trim() === ""
                  }
                >
                  {!writeLocked ? "Save assertion" : "Retry save"}
                </Button>
                {methods.map((method) => (
                  <Button
                    key={method}
                    disabled={
                      busy ||
                      writeLocked ||
                      checkLocked ||
                      conflict ||
                      statement.trim() === "" ||
                      grounds.trim() === ""
                    }
                    onClick={() => void save(method)}
                  >
                    Save and check{methods.length > 1 ? ` · ${method}` : ""}
                  </Button>
                ))}
                {editing === undefined ? null : (
                  <Button
                    disabled={busy || writeLocked}
                    onClick={() => choose()}
                  >
                    Cancel edit
                  </Button>
                )}
                {!checkLocked ? null : (
                  <Button
                    disabled={busy}
                    onClick={() => {
                      const request = checking.current;
                      if (request !== undefined)
                        void check(request.id, request.method);
                    }}
                  >
                    Retry accepting check
                  </Button>
                )}
              </div>
            </form>
          ) : null}
          {error === undefined ? null : <ErrorNotice error={error} />}
          {message === "" ? null : <p role="status">{message}</p>}
        </>
      ) : null}
    </section>
  );
}
