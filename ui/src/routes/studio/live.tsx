import {
  useInfiniteQuery,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router";
import {
  auditPollInterval,
  AUDIT_ID_PATTERN,
  getAudit,
  getAuditProfile,
  getAuditWorkspace,
  listAuditItems,
  listAuditReviews,
  listOwnerAudits,
  type Audit,
  type AuditItemPage,
  type AuditReviewPage,
} from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import type { PublicAPI } from "../../api/client";
import { queryKeys } from "../../api/query-keys";
import { ErrorNotice } from "../../app/error-notice";
import { StatusChip } from "../../ui";
import { checkLinks, checkPath } from "../projects/audits/check-links";
import { CheckEventHistory } from "../projects/audits/event-history";
import { checkStateLabel } from "../../app/vocabulary";
import { at, type Draft } from "./document";

const changed = () =>
  new Error(
    "The check changed while loading this view. Refresh to load its current revision.",
  );
function sameRevision(before: Audit, after: Audit) {
  return (
    before.revision === after.revision &&
    before.currentRoundId === after.currentRoundId
  );
}
type Position = { cursor: string; seen: string[] } | undefined;
function checkedPage<
  T extends { page: { hasMore: boolean; nextCursor?: string } },
>(page: T, position?: Position): T {
  if (
    page.page.hasMore &&
    (!page.page.nextCursor || position?.seen.includes(page.page.nextCursor))
  )
    throw new Error(
      "The server returned inconsistent pagination. Refresh this view to retry.",
    );
  return page;
}
function nextPosition(
  last: { page: { hasMore: boolean; nextCursor?: string } },
  previous: Position,
): Position {
  return last.page.hasMore
    ? {
        cursor: last.page.nextCursor!,
        seen: [...(previous?.seen ?? []), last.page.nextCursor!],
      }
    : undefined;
}

export function StudioLive({
  initialAudit,
  draft,
}: {
  initialAudit: string;
  draft: Draft;
}) {
  const api = usePublicAPI(),
    cache = useQueryClient();
  const [input, setInput] = useState(initialAudit),
    [auditId, setAuditId] = useState(
      AUDIT_ID_PATTERN.test(initialAudit) ? initialAudit : "",
    );
  const [inputError, setInputError] = useState("");
  const choices = useInfiniteQuery({
    queryKey: ["studio", "owner-checks"],
    initialPageParam: undefined as Position,
    queryFn: async ({ pageParam, signal }) =>
      checkedPage(
        await listOwnerAudits(
          api,
          pageParam ? { cursor: pageParam.cursor } : {},
          signal,
        ),
        pageParam,
      ),
    getNextPageParam: (last, _pages, previous) => nextPosition(last, previous),
    retry: false,
  });
  const snapshot = useQuery({
    queryKey: ["studio", "live", auditId],
    enabled: !!auditId,
    retry: false,
    queryFn: async () => {
      const audit = await getAudit(api, auditId);
      const [workspace, coverage, reviews, definition] = await Promise.all([
        getAuditWorkspace(api, auditId),
        listAuditItems(api, auditId),
        listAuditReviews(api, auditId, { auditRevision: audit.revision }),
        cache
          .fetchQuery({
            queryKey: queryKeys.auditProfiles.detail(
              audit.profile.name,
              audit.profile.version,
            ),
            queryFn: ({ signal }) =>
              getAuditProfile(
                api,
                audit.profile.name,
                audit.profile.version,
                signal,
              ),
            staleTime: Infinity,
          })
          .then((profile) => ({
            profile:
              profile.ref.digest === audit.profile.digest ? profile : undefined,
            error:
              profile.ref.digest === audit.profile.digest
                ? undefined
                : new Error(
                    "The catalog profile has a different digest from the definition pinned by this check. Its role definitions cannot be overlaid safely.",
                  ),
          }))
          .catch((error: unknown) => ({
            profile: undefined,
            error:
              error instanceof Error
                ? error
                : new Error("The pinned definition could not be loaded."),
          })),
      ]);
      const after = await getAudit(api, auditId);
      if (
        !sameRevision(audit, after) ||
        workspace.auditRevision !== audit.revision ||
        workspace.roundId !== audit.currentRoundId ||
        reviews.auditRevision !== audit.revision
      )
        throw changed();
      return {
        audit,
        workspace,
        coverage: checkedPage(coverage),
        reviews: checkedPage(reviews),
        profile: definition.profile,
        definitionError: definition.error,
      };
    },
    refetchInterval: (query) =>
      query.state.data
        ? auditPollInterval([query.state.data.audit], 5_000)
        : false,
    refetchOnWindowFocus: false,
  });
  const data = snapshot.data,
    audits = choices.data?.pages.flatMap((page) => page.items) ?? [];
  const matches =
    !!data &&
    draft.kind === "AuditProfile" &&
    at(draft.value, ["metadata", "name"]) === data.audit.profile.name &&
    at(draft.value, ["metadata", "version"]) === data.audit.profile.version;
  return (
    <section className="studio-live" aria-label="Live check overlay">
      <div className="studio-live-heading">
        <div>
          <h2>Live check</h2>
          <p>
            Read-only execution view. Your Design draft is kept separately in
            memory.
          </p>
        </div>
        <form
          onSubmit={(event) => {
            event.preventDefault();
            if (!AUDIT_ID_PATTERN.test(input)) {
              setInputError("Enter a valid check ID.");
              return;
            }
            setInputError("");
            setAuditId(input);
          }}
        >
          <label className="studio-field">
            <span>Check ID</span>
            <input
              value={input}
              onChange={(event) => setInput(event.target.value)}
              list="studio-check-choices"
            />
          </label>
          <datalist id="studio-check-choices">
            {audits.map((audit) => (
              <option key={audit.auditId} value={audit.auditId}>
                {audit.profile.name} · {audit.state}
              </option>
            ))}
          </datalist>
          <button className="ui-btn">Load check</button>
        </form>
      </div>
      {inputError ? <p role="alert">{inputError}</p> : null}
      {choices.error ? (
        <ErrorNotice
          error={choices.error}
          context="Check choices could not be loaded."
          onRetry={() => void choices.refetch()}
        />
      ) : null}
      {choices.hasNextPage ? (
        <button
          className="ui-btn"
          disabled={choices.isFetching}
          onClick={() => void choices.fetchNextPage()}
        >
          Load more check choices
        </button>
      ) : null}
      {snapshot.error ? (
        <ErrorNotice
          error={snapshot.error}
          context="Live view could not be refreshed. Any visible data is from the last successful snapshot."
          onRetry={() => void snapshot.refetch()}
        />
      ) : null}
      {snapshot.isFetching ? <p role="status">Refreshing live view…</p> : null}
      {!auditId ? (
        <p>Choose an existing check to see its roles, coverage and activity.</p>
      ) : null}
      {data ? (
        <>
          <p className="studio-notice">
            Pinned execution definition: {data.audit.profile.name}@
            {data.audit.profile.version} · revision {data.audit.revision}.{" "}
            {matches
              ? "The draft has the same name and version; local edits are not part of this execution."
              : "This check uses a different definition from the Design draft."}{" "}
            {data.profile
              ? "The pinned digest was verified."
              : "The pinned definition is unavailable; only execution state is shown."}
          </p>
          {data.definitionError ? (
            <ErrorNotice
              error={data.definitionError}
              context="The pinned definition is unavailable. Execution data below remains readable."
              onRetry={() => {
                void cache
                  .invalidateQueries({
                    queryKey: queryKeys.auditProfiles.detail(
                      data.audit.profile.name,
                      data.audit.profile.version,
                    ),
                  })
                  .then(() => snapshot.refetch());
              }}
            />
          ) : null}
          <div className="studio-live-toolbar">
            <StatusChip tone={checkStateLabel(data.audit.state).tone}>
              {checkStateLabel(data.audit.state).label}
            </StatusChip>
            <Link to={checkPath(data.audit.projectId, auditId)}>
              Open check
            </Link>
            <button
              className="ui-btn"
              disabled={snapshot.isFetching}
              onClick={() => void snapshot.refetch()}
            >
              Refresh live view
            </button>
            <span>Snapshot: {data.workspace.asOf}</span>
          </div>
          <div className="studio-lanes">
            <section>
              <h3>Check</h3>
              <div className="studio-live-cards">
                <div>
                  <strong>{data.audit.phase}</strong>
                  <span>Current phase</span>
                </div>
                <div>
                  <strong>
                    {data.workspace.completedChecks} /{" "}
                    {data.workspace.totalChecks}
                  </strong>
                  <span>Completed checks</span>
                </div>
                <div>
                  <strong>{data.workspace.outstandingRuns}</strong>
                  <span>Outstanding runs</span>
                </div>
                <div>
                  <strong>{data.workspace.gaps}</strong>
                  <span>Gaps</span>
                </div>
              </div>
            </section>
            <section>
              <h3>Workflow roles</h3>
              <div className="studio-live-cards">
                {Object.entries(data.profile?.workflows ?? {}).map(
                  ([name, role]) => (
                    <div key={name}>
                      <strong>{name}</strong>
                      <span>
                        {role.kind} · {role.workflow.name}@
                        {role.workflow.version}
                      </span>
                      <span>
                        {data.audit.preparation?.roles[name]?.status ??
                          "Execution details in check activity"}
                      </span>
                    </div>
                  ),
                )}
              </div>
              {!data.profile ? (
                <p>
                  Role definitions are unavailable for this pinned digest.{" "}
                  {Object.entries(data.audit.preparation?.roles ?? {}).map(
                    ([name, role]) => (
                      <span key={name}>
                        {name}: {role.status}.{" "}
                      </span>
                    ),
                  )}
                </p>
              ) : null}
            </section>
            <section>
              <h3>Reviews</h3>
              <p>
                {data.workspace.pendingReviews} pending reviews ·{" "}
                {data.workspace.unreviewedFindings} unreviewed issues
              </p>
              <LiveReviews
                key={`${auditId}:${data.audit.revision}`}
                audit={data.audit}
                first={data.reviews}
              />
            </section>
          </div>
          <LiveCoverage
            key={`${auditId}:${data.audit.revision}:${data.audit.currentRoundId}`}
            audit={data.audit}
            first={data.coverage}
            api={api}
          />
          <section className="studio-live-events">
            <h3>Event stream</h3>
            <CheckEventHistory
              audit={data.audit}
              now={undefined}
              links={checkLinks(
                data.audit.projectId,
                auditId,
                new URLSearchParams(),
                "overview",
              )}
            />
          </section>
        </>
      ) : null}
    </section>
  );
}

function LiveCoverage({
  audit,
  first,
  api,
}: {
  audit: Audit;
  first: AuditItemPage;
  api: PublicAPI;
}) {
  const query = useInfiniteQuery({
    queryKey: [
      "studio",
      "coverage",
      audit.auditId,
      audit.revision,
      audit.currentRoundId,
    ],
    initialPageParam: undefined as Position,
    initialData: { pages: [first], pageParams: [undefined] },
    staleTime: Infinity,
    queryFn: async ({ pageParam }) => {
      if (!sameRevision(audit, await getAudit(api, audit.auditId)))
        throw changed();
      const page = await listAuditItems(api, audit.auditId, {
        ...(pageParam ? { cursor: pageParam.cursor } : {}),
      });
      if (!sameRevision(audit, await getAudit(api, audit.auditId)))
        throw changed();
      return checkedPage(page, pageParam);
    },
    getNextPageParam: (last, _pages, previous) => nextPosition(last, previous),
    retry: false,
  });
  const rows = query.data.pages.flatMap((page) => page.items);
  return (
    <section>
      <h3>Item matrix</h3>
      <p>
        All rounds · {rows.length} items loaded. Current round:{" "}
        {audit.currentRoundId ?? "not started"}. Outlined items need attention;
        each cell also names its state.
      </p>
      {query.error ? (
        <ErrorNotice
          error={query.error}
          context="More items could not be loaded."
          onRetry={() => void query.fetchNextPage()}
        />
      ) : null}
      <div className="studio-matrix">
        {rows.map((row) => (
          <Link
            key={row.itemId}
            to={checkLinks(
              audit.projectId,
              audit.auditId,
              new URLSearchParams(),
              "coverage",
            ).item(row.itemId)}
            data-attention={row.finalDisposition !== "accepted-result"}
            data-state={row.state}
          >
            <strong>{row.itemKey}</strong>
            <span>
              {row.state} ·{" "}
              {row.finalDisposition === "accepted-result"
                ? "accepted"
                : "not accepted"}
            </span>
            <span>
              {row.finalDisposition ?? "awaiting result"} · {row.roundId}
            </span>
          </Link>
        ))}
      </div>
      {!rows.length ? <p>No items in this check.</p> : null}
      {query.hasNextPage ? (
        <button
          className="ui-btn"
          disabled={query.isFetching}
          onClick={() => void query.fetchNextPage()}
        >
          Load more items
        </button>
      ) : null}
    </section>
  );
}

function LiveReviews({
  audit,
  first,
}: {
  audit: Audit;
  first: AuditReviewPage;
}) {
  const api = usePublicAPI();
  const query = useInfiniteQuery({
    queryKey: ["studio", "reviews", audit.auditId, audit.revision],
    initialPageParam: undefined as Position,
    initialData: { pages: [first], pageParams: [undefined] },
    staleTime: Infinity,
    queryFn: async ({ pageParam }) => {
      const page = await listAuditReviews(api, audit.auditId, {
        auditRevision: audit.revision,
        ...(pageParam ? { cursor: pageParam.cursor } : {}),
      });
      if (page.auditRevision !== audit.revision || page.total !== first.total)
        throw changed();
      return checkedPage(page, pageParam);
    },
    getNextPageParam: (last, _pages, previous) => nextPosition(last, previous),
    retry: false,
  });
  const rows = query.data.pages.flatMap((page) => page.items);
  return (
    <>
      {query.error ? (
        <ErrorNotice
          error={query.error}
          context="More reviews could not be loaded."
          onRetry={() => void query.fetchNextPage()}
        />
      ) : null}
      <ul className="studio-reviews">
        {rows.map((row) => (
          <li key={row.requestId}>
            <Link
              to={checkLinks(
                audit.projectId,
                audit.auditId,
                new URLSearchParams(),
                "reviews",
              ).deep("reviews", { request: row.requestId })}
            >
              {row.kind}
            </Link>{" "}
            · {row.state}
          </li>
        ))}
      </ul>
      <p>
        {rows.length} of {first.total} reviews loaded.
      </p>
      {query.hasNextPage ? (
        <button
          className="ui-btn"
          disabled={query.isFetching}
          onClick={() => void query.fetchNextPage()}
        >
          Load more reviews
        </button>
      ) : null}
    </>
  );
}
