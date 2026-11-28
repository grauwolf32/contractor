import { useHypothesisPage, useTaskPage } from "./research-pages";
import { useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { Link, useSearchParams } from "react-router";
import { type Audit } from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { PublicAPIError, requireData } from "../../../api/error";
import { readableValue } from "../../../app/vocabulary";
import { useCheckNavigation } from "./check-navigation";
import { taskStatus } from "../../../modules/checks/research";
import { ResearchTaskDetail } from "./research-work";
import type { components } from "../../../api/generated/public";
import { queryKeys } from "../../../api/query-keys";
import { CursorControls } from "../../../app/cursor-controls";
import { ErrorNotice } from "../../../app/error-notice";
import { QueryView } from "../../../app/query-view";
import { createMutationIdempotencyKey } from "../../../mutations/idempotency";
import {
  Button,
  Field,
  TechnicalDetails,
  StatusChip,
  Facts,
} from "../../../ui";

type Hypothesis = components["schemas"]["AuditHypothesis"];
type Content = components["schemas"]["AuditHypothesisContent"];
type PendingWrite = {
  method?: string;
  key: string;
  revision: number;
  content: Content;
  id?: string;
};

/** Assertions are owner commands over ordinary retained Task results. */
export function AuditHypotheses({
  audit,
  expanded = false,
  active = true,
}: {
  audit: Audit;
  expanded?: boolean;
  active?: boolean;
}) {
  const api = usePublicAPI();
  const { sectionHref } = useCheckNavigation(audit);
  const [search, setSearch] = useSearchParams();
  const cache = useQueryClient();
  const [open, setOpen] = useState(expanded);
  const [formOpen, setFormOpen] = useState(!expanded);
  const [editing, setEditing] = useState<Hypothesis>();
  const [formRevision, setFormRevision] = useState(audit.revision);
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
  const editable = audit.state === "active" || audit.state === "paused";
  const { query, pages } = useHypothesisPage(audit, open && active);
  const methods = query.data?.checking_workflows ?? [];
  const currentSource = query.data?.items.find(
    (value) => value.document.id === editing?.document.id,
  );
  const sourceChanged =
    editing !== undefined &&
    currentSource !== undefined &&
    currentSource.document.revision !== editing.document.revision;
  const formConflict = conflict || (sourceChanged && !writeLocked);

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
    setFormOpen(true);
    if (pending.current !== undefined || busy) return;
    setEditing(value);
    setFormRevision(audit.revision);
    setStatement(value?.document.statement ?? "");
    setConditions(value?.document.scope.conditions ?? "");
    setGrounds(value?.document.grounds ?? "");
    setConflict(false);
    setError(undefined);
    setMessage("");
  }

  async function save(method?: string) {
    if (formConflict && !writeLocked) return;
    const request = pending.current ?? {
      key: createMutationIdempotencyKey("hypothesis"),
      ...(method === undefined ? {} : { method }),
      revision: formRevision,
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
      let accepted: { audit_revision: number };
      if (request.method !== undefined) {
        const taskKey = request.key;
        accepted = requireData(
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
        accepted = requireData(
          await api.request((client) =>
            client.POST("/v1/audits/{auditId}/hypotheses", {
              params,
              body: request.content,
            }),
          ),
        );
      } else {
        accepted = requireData(
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
      setFormRevision(accepted.audit_revision);
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

  const chosen =
    query.data?.items.find((h) => h.document.id === search.get("hypothesis")) ??
    query.data?.items[0];
  return (
    <section className="research-hypotheses ui-stack" aria-label="Assertions">
      {!expanded ? (
        <Button
          size="sm"
          onClick={() => {
            if (
              !open &&
              editing === undefined &&
              statement === "" &&
              grounds === "" &&
              conditions === ""
            )
              setFormRevision(audit.revision);
            setOpen(!open);
          }}
        >
          {open ? "Hide assertions" : "Show assertions"}
        </Button>
      ) : null}
      {open ? (
        <>
          <header className="research-section-head">
            <div>
              <p className="eyebrow">Assertions & results</p>
              <h2>Hypotheses</h2>
            </div>
            {editable ? (
              <Button size="sm" onClick={() => choose()}>
                Add hypothesis
              </Button>
            ) : null}
          </header>
          <QueryView
            query={query}
            loading={<p role="status">Loading assertions…</p>}
          >
            {(page) => (
              <>
                <p className="ui-caption">
                  {page.fixed_denominator} requirements in the original
                  checklist. New assertions, split checks and missing evidence
                  do not change this total.
                </p>
                {page.pending_fixed_count ? (
                  <p role="status">
                    {page.pending_fixed_count} original requirements have no
                    accepted assertion yet. They remain unchecked.
                  </p>
                ) : null}
                <div className="research-hypothesis-workspace">
                  <div className="research-hypothesis-list">
                    {page.items.map((value) => (
                      <Button
                        key={value.document.id}
                        variant="ghost"
                        className="research-hypothesis-choice"
                        aria-pressed={chosen?.document.id === value.document.id}
                        onClick={() =>
                          setSearch((previous) => {
                            const next = new URLSearchParams(previous);
                            next.set("hypothesis", value.document.id);
                            next.delete("historyCursor");
                            next.delete("historyTask");
                            return next;
                          })
                        }
                      >
                        <StatusChip
                          tone={
                            value.fixed
                              ? "neutral"
                              : value.document.origin.kind === "run"
                                ? "info"
                                : "neutral"
                          }
                          size="sm"
                        >
                          {value.fixed
                            ? "Fixed"
                            : value.document.origin.kind === "run"
                              ? "Generated"
                              : "Manual"}
                        </StatusChip>
                        <strong>{value.document.statement}</strong>
                        <small>Revision {value.document.revision}</small>
                      </Button>
                    ))}
                    <CursorControls
                      label="Assertion pages"
                      {...pages.controls(page.page)}
                    />
                  </div>
                  {chosen ? (
                    <article className="research-hypothesis-detail">
                      <header>
                        <StatusChip
                          tone={chosen.fixed ? "neutral" : "info"}
                          size="sm"
                        >
                          {chosen.fixed
                            ? "Original checklist requirement"
                            : "Additional assertion"}
                        </StatusChip>
                        <h2>{chosen.document.statement}</h2>
                        <p className="ui-caption">
                          Exact assertion · revision {chosen.document.revision}
                        </p>
                      </header>
                      <section>
                        <h3>Scope & conditions</h3>
                        <Facts
                          layout="list"
                          items={Object.entries(chosen.document.scope).map(
                            ([term, value]) => ({
                              term: readableValue(term),
                              value,
                            }),
                          )}
                        />
                      </section>
                      <section>
                        <h3>Why investigate this</h3>
                        <p>{chosen.document.grounds}</p>
                      </section>
                      <section>
                        <h3>Grounds · exact materials</h3>
                        {chosen.document.sources?.length ? (
                          chosen.document.sources.map((source) => (
                            <Link
                              className="research-source-link"
                              key={JSON.stringify(source)}
                              to={sectionHref("materials", {
                                source: JSON.stringify(source),
                              })}
                            >
                              <span>{source.ref.name}</span>
                              <small>
                                {source.media_type} · {source.ref.revision}
                              </small>
                              <span aria-hidden="true">↗</span>
                            </Link>
                          ))
                        ) : (
                          <p className="ui-caption">
                            No exact source material was selected.
                          </p>
                        )}
                      </section>
                      <section>
                        <h3>Origin</h3>
                        <p>
                          {chosen.document.origin.kind === "run"
                            ? "Generated by a check workflow"
                            : chosen.document.origin.kind === "checklist"
                              ? "Frozen checklist source"
                              : "Explicit analyst input"}
                        </p>
                        {chosen.document.origin.run_id ? (
                          <Link
                            to={`/runs/${encodeURIComponent(chosen.document.origin.run_id)}`}
                          >
                            Recorded source Run →
                          </Link>
                        ) : null}
                      </section>
                      <div className="ui-toolbar">
                        {editable ? (
                          <Button
                            size="sm"
                            disabled={busy || writeLocked || checkLocked}
                            onClick={() => choose(chosen)}
                          >
                            Edit assertion
                          </Button>
                        ) : null}
                        {editable
                          ? methods.map((method) => (
                              <Button
                                key={method}
                                size="sm"
                                disabled={busy || writeLocked || checkLocked}
                                onClick={() =>
                                  void check(chosen.document.id, method)
                                }
                              >
                                Check assertion
                                {methods.length > 1 ? ` · ${method}` : ""}
                              </Button>
                            ))
                          : null}
                        <Link
                          to={sectionHref("work", {
                            hypothesis: chosen.document.id,
                          })}
                        >
                          View checks and results →
                        </Link>
                      </div>
                      {active ? (
                        <HypothesisHistory
                          audit={audit}
                          hypothesis={chosen.document.id}
                        />
                      ) : null}
                      <TechnicalDetails description="Exact revision, origin and evidence requirements.">
                        <pre className="checks-task-json">
                          {JSON.stringify(chosen, null, 2)}
                        </pre>
                      </TechnicalDetails>
                    </article>
                  ) : (
                    <p>
                      No registered hypotheses yet. Saved proposal outputs are
                      separate from registration.
                    </p>
                  )}
                </div>
              </>
            )}
          </QueryView>
          {editable && formOpen ? (
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
              {formConflict ? (
                <p role="alert">
                  {editing === undefined
                    ? "This check changed. Review your new assertion and use the refreshed check revision before retrying."
                    : "This assertion or check changed. Your draft is preserved. Select Edit assertion again to load its current content before saving."}
                </p>
              ) : null}
              <div className="ui-toolbar">
                {conflict && editing === undefined ? (
                  <Button
                    disabled={busy}
                    onClick={() => {
                      setFormRevision(audit.revision);
                      setConflict(false);
                    }}
                  >
                    Use refreshed revision
                  </Button>
                ) : null}
                <Button
                  type="submit"
                  disabled={
                    busy ||
                    formConflict ||
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
                      formConflict ||
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

function HypothesisHistory({
  audit,
  hypothesis,
}: {
  audit: Audit;
  hypothesis: string;
}) {
  const { query, pages } = useTaskPage(audit, hypothesis);
  const [historyParams, setHistoryParams] = useSearchParams();
  const selected = historyParams.get("historyTask");
  return (
    <section className="research-hypothesis-history">
      <h3>Check history</h3>
      <QueryView query={query}>
        {(page) => {
          const chosen =
            page.items.find((t) => t.document.task_id === selected) ??
            page.items[0];
          return (
            <>
              {page.items.length ? (
                <>
                  <div className="ui-toolbar">
                    {page.items.map((t) => (
                      <Button
                        size="sm"
                        key={t.document.task_id}
                        aria-pressed={
                          chosen?.document.task_id === t.document.task_id
                        }
                        onClick={() =>
                          setHistoryParams(
                            (previous) => {
                              const next = new URLSearchParams(previous);
                              next.set("historyTask", t.document.task_id);
                              return next;
                            },
                            { preventScrollReset: true },
                          )
                        }
                      >
                        Revision {t.document.hypothesis?.revision} ·{" "}
                        {taskStatus(t).label}
                      </Button>
                    ))}
                  </div>
                  {chosen ? (
                    <ResearchTaskDetail
                      key={chosen.document.task_id}
                      audit={audit}
                      task={chosen}
                    />
                  ) : null}
                </>
              ) : (
                <p className="ui-caption">
                  No checking task is recorded. Registration does not schedule a
                  check automatically.
                </p>
              )}
              <CursorControls
                label="Hypothesis check pages"
                {...pages.controls(page.page)}
              />
            </>
          );
        }}
      </QueryView>
    </section>
  );
}
