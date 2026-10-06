import { useQuery } from "@tanstack/react-query";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import {
  getAuditFinding,
  type AuditAnalystVerdict,
  type AuditFinding,
  type AuditReviewRequest,
} from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import { FindingDecision, type FindingDecisionProps } from "./finding-decision";
import {
  AUDIT_ID,
  failure,
  FINDING_ID,
  json,
  makeDecision,
  makeFinding,
  makeTriageReview,
  NOW,
  renderWithServer,
  type Handler,
} from "./test-support";

const STATE_AFTER: Record<AuditAnalystVerdict, AuditFinding["state"]> = {
  true_positive: "confirmed",
  false_positive: "rejected",
  duplicate: "duplicate",
  needs_evidence: "needs-evidence",
  reopen: "proposed",
};

interface FakeOptions {
  finding: AuditFinding;
  pending?: AuditReviewRequest[];
  decided?: AuditReviewRequest[];
  others?: AuditFinding[];
  /** Answers the list of the check's possible issues instead of the Server. */
  listFindings?: () => Response | Promise<Response>;
  /** Answers the open-request lookup, when it returns a response. */
  pendingLookup?: () => Response | undefined;
  /** Answers a decision instead of recording it, when it returns a response. */
  refuseDecision?: (
    attempt: number,
    state: ServerState,
  ) => Response | undefined;
}

interface ServerState {
  finding: AuditFinding;
  pending: AuditReviewRequest[];
  decided: AuditReviewRequest[];
  findingReads: number;
  decisionAttempts: number;
}

/** A Server that keeps one possible issue and its review requests. */
function fakeServer(options: FakeOptions) {
  const state: ServerState = {
    finding: options.finding,
    pending: options.pending ?? [],
    decided: options.decided ?? [],
    findingReads: 0,
    decisionAttempts: 0,
  };
  const page = (items: unknown[]) =>
    json({
      items,
      page: { hasMore: false },
      total: items.length,
      auditRevision: 1,
      asOf: NOW,
    });
  const handle: Handler = async (request, url) => {
    const path = url.pathname;
    const base = `/v1/audits/${AUDIT_ID}`;
    if (request.method === "GET" && path === `${base}/findings/${FINDING_ID}`) {
      state.findingReads += 1;
      return json(state.finding, 200, {
        ETag: `"${state.finding.revision}"`,
      });
    }
    if (request.method === "GET" && path === `${base}/findings`)
      return (
        options.listFindings?.() ??
        page([state.finding, ...(options.others ?? [])])
      );
    if (request.method === "GET" && path === `${base}/reviews`) {
      expect(url.searchParams.get("finding")).toBe(FINDING_ID);
      if (url.searchParams.get("state") === "pending") {
        const answer = options.pendingLookup?.();
        if (answer !== undefined) return answer;
      }
      return page(
        url.searchParams.get("state") === "decided"
          ? state.decided
          : state.pending,
      );
    }
    if (
      request.method === "POST" &&
      path === `${base}/findings/${FINDING_ID}/reviews`
    ) {
      const created = makeTriageReview({
        requestId: "review_new",
        subjectRevision: state.finding.revision,
        revision: 1,
      });
      state.pending = [created];
      return json(created, 201, { ETag: '"1"' });
    }
    const decision = new RegExp(`^${base}/reviews/([^/]+)/decisions$`).exec(
      path,
    );
    if (request.method === "POST" && decision !== null) {
      state.decisionAttempts += 1;
      const refused = options.refuseDecision?.(state.decisionAttempts, state);
      if (refused !== undefined) return refused;
      const body = (await request.clone().json()) as {
        verdict: AuditAnalystVerdict;
        severity?: "high";
        rationale: string;
        duplicateTargetId?: string;
      };
      const review = state.pending.find(
        (candidate) => candidate.requestId === decision[1],
      );
      if (review === undefined)
        return failure(404, "not_found", "review not found");
      const recorded = makeDecision({
        requestId: review.requestId,
        verdict: body.verdict,
        rationale: body.rationale,
        subjectRevision: state.finding.revision,
        ...(body.severity === undefined ? {} : { severity: body.severity }),
        ...(body.duplicateTargetId === undefined
          ? {}
          : { duplicateTargetId: body.duplicateTargetId }),
      });
      if (body.severity === undefined) delete recorded.severity;
      // Only true and false positives carry an effective analyst decision.
      const next: AuditFinding = {
        ...state.finding,
        state: STATE_AFTER[body.verdict],
        revision: state.finding.revision + 1,
      };
      delete next.analystDecision;
      delete next.analystVerdict;
      delete next.analystSeverity;
      delete next.duplicateTargetId;
      if (
        body.verdict === "true_positive" ||
        body.verdict === "false_positive"
      ) {
        next.analystDecision = recorded;
        next.analystVerdict = body.verdict;
        if (body.severity !== undefined) next.analystSeverity = body.severity;
      }
      if (body.duplicateTargetId !== undefined)
        next.duplicateTargetId = body.duplicateTargetId;
      state.finding = next;
      const decided: AuditReviewRequest = {
        ...review,
        state: "decided",
        revision: review.revision + 1,
        decision: recorded,
      };
      state.pending = [];
      state.decided = [...state.decided, decided];
      return json(
        {
          finding: state.finding,
          request: decided,
          decision: recorded,
          replayed: false,
        },
        200,
        { ETag: `"${decided.revision}"` },
      );
    }
    throw new Error(`unexpected ${request.method} ${path}${url.search}`);
  };
  return { state, handle };
}

/** Reads the possible issue like a page does and decides on it. */
function LiveDecision(
  props: Omit<FindingDecisionProps, "auditId" | "finding">,
) {
  const api = usePublicAPI();
  const finding = useQuery({
    queryKey: [
      ...queryKeys.audits.detail(AUDIT_ID),
      "findings",
      FINDING_ID,
      "exact",
    ],
    queryFn: () => getAuditFinding(api, AUDIT_ID, FINDING_ID),
  });
  return finding.data === undefined ? (
    <p>Loading the possible issue</p>
  ) : (
    <FindingDecision auditId={AUDIT_ID} finding={finding.data} {...props} />
  );
}

function setup(
  options: FakeOptions,
  props: Omit<FindingDecisionProps, "auditId" | "finding"> = {},
) {
  const server = fakeServer(options);
  const view = renderWithServer(<LiveDecision {...props} />, server.handle);
  return { ...server, ...view, user: userEvent.setup() };
}

async function decisionBar() {
  return screen.findByRole("region", { name: "Your decision" });
}

/** The bar is shown and the review status has loaded. */
async function readyBar() {
  const bar = await decisionBar();
  await waitFor(() =>
    expect(
      within(bar).getAllByRole("button", { pressed: false })[0],
    ).toBeEnabled(),
  );
  return bar;
}

function reasonField() {
  return screen.getByRole("textbox", { name: "Why" });
}

function recordButton() {
  return screen.getByRole("button", { name: "Record decision" });
}

async function bodyOf(request: Request | undefined): Promise<unknown> {
  if (request === undefined) throw new Error("request was not sent");
  return request.clone().json();
}

describe("FindingDecision", () => {
  it("opens a review for the exact revision, then confirms with the chosen severity", async () => {
    const onDecided = vi.fn();
    const { user, sent } = setup({ finding: makeFinding() }, { onDecided });
    await decisionBar();
    await waitFor(() =>
      expect(recordButton()).toHaveAccessibleDescription("Choose a decision."),
    );
    // The AI's suggestion is shown as a suggestion and never preselected.
    expect(
      screen.getByText("Not set yet. The AI suggests High."),
    ).toBeVisible();
    expect(screen.getByRole("radio", { name: "High" })).not.toBeChecked();

    await user.click(screen.getByRole("button", { name: "Confirm issue" }));
    expect(recordButton()).toBeDisabled();
    expect(recordButton()).toHaveAccessibleDescription("Choose severity.");
    await user.click(screen.getByRole("radio", { name: "High" }));
    expect(recordButton()).toHaveAccessibleDescription("Write a short reason.");
    await user.type(reasonField(), "  Confirmed against the trace.  ");
    await user.click(recordButton());

    const region = await screen.findByRole("region", {
      name: "Current decision",
    });
    expect(within(region).getByText("Confirmed · High")).toBeVisible();
    expect(within(region).getByText("user_analyst")).toBeVisible();
    expect(
      within(region).getByRole("button", { name: "Change decision" }),
    ).toBeVisible();
    expect(
      screen.queryByRole("region", { name: "Your decision" }),
    ).not.toBeInTheDocument();

    const [create] = sent("POST", `/findings/${FINDING_ID}/reviews`);
    expect(create?.headers.get("If-Match")).toBe('"3"');
    expect(create?.headers.get("Idempotency-Key")).toMatch(
      /^audit-finding-review-ui-/u,
    );
    expect(await bodyOf(create)).toEqual({});
    const decisions = sent("POST", "/reviews/review_new/decisions");
    expect(decisions).toHaveLength(1);
    expect(decisions[0]?.headers.get("If-Match")).toBe('"1"');
    expect(decisions[0]?.headers.get("Idempotency-Key")).toMatch(
      /^audit-finding-review-ui-/u,
    );
    expect(await bodyOf(decisions[0])).toEqual({
      verdict: "true_positive",
      severity: "high",
      rationale: "Confirmed against the trace.",
    });
    expect(onDecided).toHaveBeenCalledTimes(1);
    expect(onDecided.mock.calls[0]?.[0]).toMatchObject({
      decision: { verdict: "true_positive", severity: "high" },
    });
  });

  it.each([
    ["Not an issue", "false_positive"],
    ["Needs evidence", "needs_evidence"],
  ] as const)("records %s without a severity", async (label, verdict) => {
    const { user, sent } = setup({ finding: makeFinding() });
    await readyBar();
    // A severity chosen before switching verdicts is not sent.
    await user.click(screen.getByRole("radio", { name: "Low" }));
    await user.click(screen.getByRole("button", { name: label }));
    expect(screen.getByText("Saved only when you confirm.")).toBeVisible();
    await user.type(reasonField(), "Reviewed by hand.");
    await user.click(recordButton());
    await waitFor(() =>
      expect(sent("POST", "/reviews/review_new/decisions")).toHaveLength(1),
    );
    expect(
      await bodyOf(sent("POST", "/reviews/review_new/decisions")[0]),
    ).toEqual({ verdict, rationale: "Reviewed by hand." });
  });

  it("marks a duplicate of another possible issue chosen by search or by exact ID", async () => {
    const other = makeFinding(
      { findingId: "finding_2" },
      { title: "Missing rate limit" },
    );
    const { user, sent } = setup({ finding: makeFinding(), others: [other] });
    await decisionBar();
    await user.click(screen.getByRole("button", { name: "More decisions" }));
    // Reopen is only for decided possible issues.
    expect(screen.queryByRole("button", { name: "Reopen" })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Duplicate…" }));
    const picker = screen.getByRole("group", { name: "Duplicate of" });
    const search = within(picker).getByRole("searchbox", {
      name: "Search possible issues in this check",
    });
    await waitFor(() => expect(search).toHaveFocus());
    expect(screen.getByRole("button", { name: "Duplicate" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );

    // Recording without an original explains what is missing.
    await user.type(reasonField(), "Same root cause.");
    await user.click(recordButton());
    expect(
      screen.getByText("Choose the possible issue this one duplicates."),
    ).toBeVisible();
    expect(search).toHaveFocus();
    expect(sent("POST", "/reviews")).toHaveLength(0);

    // The current possible issue is not offered as its own original.
    const option = await within(picker).findByRole("radio", {
      name: "Missing rate limit finding_2",
    });
    expect(within(picker).getAllByRole("radio")).toHaveLength(1);
    await user.type(search, "nothing like it");
    expect(within(picker).queryAllByRole("radio")).toHaveLength(0);
    await user.clear(search);
    await user.type(search, "finding_77");
    await user.click(
      within(picker).getByRole("radio", { name: "Use this ID finding_77" }),
    );
    expect(
      within(picker).getByRole("radio", {
        name: "ID entered by you finding_77",
      }),
    ).toBeChecked();
    await user.clear(search);
    const sibling = within(picker).getByRole("radio", {
      name: "Missing rate limit finding_2",
    });
    await user.click(sibling);
    expect(sibling).toBeChecked();
    expect(option).not.toBeInTheDocument();
    await user.click(recordButton());

    await waitFor(() =>
      expect(sent("POST", "/reviews/review_new/decisions")).toHaveLength(1),
    );
    expect(
      await bodyOf(sent("POST", "/reviews/review_new/decisions")[0]),
    ).toEqual({
      verdict: "duplicate",
      duplicateTargetId: "finding_2",
      rationale: "Same root cause.",
    });
  });

  it("marks a duplicate by the exact ID when the possible issues cannot be listed", async () => {
    // The 5-page list is revision-fenced: an active check can answer 409.
    const { user, sent } = setup({
      finding: makeFinding(),
      listFindings: () =>
        failure(409, "conflict", "audit revision changed while paging"),
    });
    await readyBar();
    await user.click(screen.getByRole("button", { name: "More decisions" }));
    await user.click(screen.getByRole("button", { name: "Duplicate…" }));
    const picker = screen.getByRole("group", { name: "Duplicate of" });
    expect(
      await within(picker).findByText(
        /The possible issues of this check could not be loaded\. You can still enter the exact ID of the original\./u,
      ),
    ).toBeVisible();
    // Not the possible issue itself, and not a malformed ID.
    const search = within(picker).getByRole("searchbox", {
      name: "Search possible issues in this check",
    });
    await user.type(search, FINDING_ID);
    expect(within(picker).queryAllByRole("radio")).toHaveLength(0);
    await user.clear(search);
    await user.type(search, "finding 77");
    expect(within(picker).queryAllByRole("radio")).toHaveLength(0);
    await user.clear(search);
    await user.type(search, "finding_77");
    await user.click(
      within(picker).getByRole("radio", { name: "Use this ID finding_77" }),
    );
    expect(
      within(picker).getByRole("radio", {
        name: "ID entered by you finding_77",
      }),
    ).toBeChecked();
    await user.type(reasonField(), "Same handler as finding 77.");
    await user.click(recordButton());

    const region = await screen.findByRole("region", {
      name: "Current decision",
    });
    expect(within(region).getByText("Duplicate")).toBeVisible();
    expect(
      await bodyOf(sent("POST", "/reviews/review_new/decisions")[0]),
    ).toEqual({
      verdict: "duplicate",
      duplicateTargetId: "finding_77",
      rationale: "Same handler as finding 77.",
    });
  });

  it("offers the exact ID while the possible issues are still loading", async () => {
    let release: (() => void) | undefined;
    const listed = new Promise<void>((resolve) => {
      release = resolve;
    });
    const { user, sent } = setup({
      finding: makeFinding(),
      listFindings: async () => {
        await listed;
        return json({
          items: [],
          page: { hasMore: false },
          total: 0,
          auditRevision: 1,
          asOf: NOW,
        });
      },
    });
    await readyBar();
    await user.click(screen.getByRole("button", { name: "More decisions" }));
    await user.click(screen.getByRole("button", { name: "Duplicate…" }));
    const picker = screen.getByRole("group", { name: "Duplicate of" });
    expect(within(picker).getByText(/Loading possible issues…/u)).toBeVisible();
    await user.type(
      within(picker).getByRole("searchbox", {
        name: "Search possible issues in this check",
      }),
      "finding_77",
    );
    await user.click(
      within(picker).getByRole("radio", { name: "Use this ID finding_77" }),
    );
    await user.type(reasonField(), "Same handler as finding 77.");
    await user.click(recordButton());
    await waitFor(() =>
      expect(sent("POST", "/reviews/review_new/decisions")).toHaveLength(1),
    );
    expect(
      await bodyOf(sent("POST", "/reviews/review_new/decisions")[0]),
    ).toMatchObject({ verdict: "duplicate", duplicateTargetId: "finding_77" });
    release?.();
    expect(
      await screen.findByRole("region", { name: "Current decision" }),
    ).toBeVisible();
  });

  it("announces the recorded decision after the refetch until the next action", async () => {
    const { user } = setup({ finding: makeFinding() });
    await readyBar();
    await user.click(screen.getByRole("button", { name: "Not an issue" }));
    await user.type(reasonField(), "Guarded by the gateway.");
    await user.click(recordButton());

    const region = await screen.findByRole("region", {
      name: "Current decision",
    });
    expect(within(region).getByText("Not an issue")).toBeVisible();
    // The refetched possible issue replaced the bar; the live region still
    // says what was recorded, and focus stays on this decision.
    expect(screen.getByRole("status")).toHaveTextContent(
      "Decision recorded: Not an issue.",
    );
    expect(
      screen.getByRole("group", {
        name: "Decision on Missing object authorization",
      }),
    ).toHaveFocus();
    await user.click(
      within(region).getByRole("button", { name: "Change decision" }),
    );
    expect(screen.getByRole("status")).toBeEmptyDOMElement();
  });

  it("explains a review status that cannot be read, with its request details", async () => {
    let lookupFails = true;
    const { user } = setup({
      finding: makeFinding(),
      pendingLookup: () =>
        lookupFails
          ? failure(503, "unavailable", "review store unavailable")
          : undefined,
    });
    const bar = await decisionBar();
    await waitFor(() =>
      expect(recordButton()).toHaveAccessibleDescription(
        "The review status could not be loaded.",
      ),
    );
    expect(within(bar).getByRole("alert")).toHaveTextContent(
      "review store unavailable",
    );
    expect(
      screen.getByRole("button", { name: "Confirm issue" }),
    ).toBeDisabled();
    await user.click(screen.getByText("Request details"));
    expect(screen.getByText("Code unavailable · Status 503")).toBeVisible();
    expect(screen.getByText("Request req_1")).toBeVisible();
    // The message is already on screen; the details do not repeat it.
    expect(screen.queryByText(/^Message:/u)).toBeNull();

    lookupFails = false;
    await user.click(within(bar).getByRole("button", { name: "Try again" }));
    await readyBar();
    expect(screen.queryByText("Request details")).toBeNull();
  });

  it("reopens a decided possible issue through Change decision", async () => {
    const decided = makeFinding({
      state: "confirmed",
      revision: 5,
      analystDecision: makeDecision(),
      analystVerdict: "true_positive",
      analystSeverity: "high",
    });
    const { user, sent } = setup({ finding: decided });
    const region = await screen.findByRole("region", {
      name: "Current decision",
    });
    expect(within(region).getByText("Confirmed · High")).toBeVisible();
    expect(
      await within(region).findByText("retained", { selector: "strong" }),
    ).toBeVisible();
    expect(
      screen.queryByRole("region", { name: "Your decision" }),
    ).not.toBeInTheDocument();

    await user.click(
      within(region).getByRole("button", { name: "Change decision" }),
    );
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Confirm issue" }),
      ).toHaveFocus(),
    );
    // The analyst's own severity stays chosen.
    expect(screen.getByRole("radio", { name: "High" })).toBeChecked();
    await user.click(screen.getByRole("button", { name: "More decisions" }));
    await user.click(screen.getByRole("button", { name: "Reopen" }));
    await waitFor(() => expect(reasonField()).toHaveFocus());
    await user.type(reasonField(), "The fix was reverted.");
    await user.click(recordButton());

    await waitFor(() =>
      expect(
        screen.queryByRole("region", { name: "Current decision" }),
      ).not.toBeInTheDocument(),
    );
    expect(await decisionBar()).toBeVisible();
    const [create] = sent("POST", `/findings/${FINDING_ID}/reviews`);
    expect(create?.headers.get("If-Match")).toBe('"5"');
    expect(
      await bodyOf(sent("POST", "/reviews/review_new/decisions")[0]),
    ).toEqual({ verdict: "reopen", rationale: "The fix was reverted." });
  });

  it("can keep the current decision after opening the bar", async () => {
    const { user } = setup({
      finding: makeFinding({
        state: "rejected",
        rejectionReason: "false-positive",
        analystDecision: makeDecision({ verdict: "false_positive" }),
        analystVerdict: "false_positive",
      }),
    });
    await user.click(
      await screen.findByRole("button", { name: "Change decision" }),
    );
    expect(await decisionBar()).toBeVisible();
    await user.click(
      screen.getByRole("button", { name: "Keep current decision" }),
    );
    expect(
      screen.queryByRole("region", { name: "Your decision" }),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Change decision" }),
    ).toHaveFocus();
  });

  it("reuses the open review request instead of opening another", async () => {
    const open = makeTriageReview({ revision: 4 });
    const { user, sent } = setup({ finding: makeFinding(), pending: [open] });
    await decisionBar();
    await user.click(screen.getByRole("button", { name: "Not an issue" }));
    await user.type(reasonField(), "Owner check exists upstream.");
    await waitFor(() => expect(recordButton()).toBeEnabled());
    await user.click(recordButton());
    await waitFor(() =>
      expect(sent("POST", "/reviews/review_open/decisions")).toHaveLength(1),
    );
    expect(
      sent("POST", "/reviews/review_open/decisions")[0]?.headers.get(
        "If-Match",
      ),
    ).toBe('"4"');
    expect(sent("POST", `/findings/${FINDING_ID}/reviews`)).toHaveLength(0);
  });

  it("uses a review request the page already read without reading reviews", async () => {
    const open = makeTriageReview({
      requestedActions: ["true_positive", "false_positive"],
    });
    const { sent } = setup(
      { finding: makeFinding(), pending: [open] },
      { pendingReview: open },
    );
    await decisionBar();
    // Only the verdicts the open request offers.
    expect(screen.getByRole("button", { name: "Not an issue" })).toBeEnabled();
    expect(screen.queryByRole("button", { name: "Needs evidence" })).toBeNull();
    expect(screen.queryByRole("button", { name: "More decisions" })).toBeNull();
    expect(sent("GET", "/reviews")).toHaveLength(0);
  });

  it("explains a stale revision, refetches and records only when asked again", async () => {
    const { user, sent, state } = setup({
      finding: makeFinding(),
      refuseDecision: (attempt) =>
        attempt === 1
          ? failure(
              412,
              "precondition_failed",
              "resource revision precondition failed",
            )
          : undefined,
    });
    await decisionBar();
    await user.click(screen.getByRole("button", { name: "Not an issue" }));
    await user.type(reasonField(), "Guarded by the gateway.");
    const readsBefore = state.findingReads;
    await user.click(recordButton());

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Not saved: this possible issue or its check changed first. The latest version is now shown.",
    );
    // The request details stay one click away, as in every error notice.
    await user.click(screen.getByText("Request details"));
    expect(
      screen.getByText("Code precondition_failed · Status 412"),
    ).toBeVisible();
    expect(screen.getByText("Request req_1")).toBeVisible();
    expect(
      screen.getByText("Message: resource revision precondition failed"),
    ).toBeVisible();
    expect(state.findingReads).toBeGreaterThan(readsBefore);
    expect(sent("POST", "/decisions")).toHaveLength(1);
    // The choice and reason stay for an explicit second attempt.
    expect(reasonField()).toHaveValue("Guarded by the gateway.");
    expect(
      screen.getByRole("button", { name: "Not an issue" }),
    ).toHaveAttribute("aria-pressed", "true");

    await user.click(recordButton());
    expect(
      await screen.findByRole("region", { name: "Current decision" }),
    ).toBeVisible();
    // The review opened by the first attempt is reused.
    expect(sent("POST", `/findings/${FINDING_ID}/reviews`)).toHaveLength(1);
    expect(sent("POST", "/decisions")).toHaveLength(2);
  });

  it("keeps the explanation when the refetch shows a decision made elsewhere", async () => {
    const { user, sent } = setup({
      finding: makeFinding(),
      refuseDecision: (_attempt, state) => {
        state.finding = makeFinding({
          state: "confirmed",
          revision: 4,
          analystDecision: makeDecision({ actorId: "user_colleague" }),
          analystVerdict: "true_positive",
          analystSeverity: "high",
        });
        state.pending = [];
        return failure(
          412,
          "precondition_failed",
          "resource revision precondition failed",
        );
      },
    });
    await readyBar();
    await user.click(screen.getByRole("button", { name: "Not an issue" }));
    await user.type(reasonField(), "Behind the admin gateway.");
    await user.click(recordButton());

    const region = await screen.findByRole("region", {
      name: "Current decision",
    });
    expect(within(region).getByText("user_colleague")).toBeVisible();
    const notice = screen.getByRole("alert");
    expect(notice).toHaveTextContent(
      "Not saved: this possible issue or its check changed first.",
    );
    await user.click(within(notice).getByText("Request details"));
    expect(
      within(notice).getByText("Code precondition_failed · Status 412"),
    ).toBeVisible();
    expect(
      screen.queryByRole("region", { name: "Your decision" }),
    ).not.toBeInTheDocument();
    // Nothing is retried; the choice is still there to record on purpose.
    expect(sent("POST", "/decisions")).toHaveLength(1);
    await user.click(
      within(region).getByRole("button", { name: "Change decision" }),
    );
    expect(reasonField()).toHaveValue("Behind the admin gateway.");
  });

  it("blocks deciding while an open request is newer than the possible issue shown", async () => {
    setup({
      finding: makeFinding(),
      pending: [makeTriageReview({ subjectRevision: 4 })],
    });
    await decisionBar();
    await waitFor(() =>
      expect(recordButton()).toHaveAccessibleDescription(
        "This possible issue changed since it was loaded.",
      ),
    );
    expect(
      screen.getByRole("button", { name: "Confirm issue" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Load the latest version" }),
    ).toBeVisible();
  });

  it("chooses verdicts with C, R and E and records with Ctrl+Enter", async () => {
    const { user, sent } = setup({ finding: makeFinding() });
    await readyBar();
    expect(
      screen.getByRole("button", { name: "Confirm issue" }),
    ).toHaveAttribute("aria-keyshortcuts", "C");
    await user.keyboard("c");
    expect(
      screen.getByRole("button", { name: "Confirm issue" }),
    ).toHaveAttribute("aria-pressed", "true");
    expect(reasonField()).toHaveFocus();
    // Typing in the reason does not switch verdicts.
    await user.keyboard("e");
    expect(reasonField()).toHaveValue("e");
    expect(
      screen.getByRole("button", { name: "Confirm issue" }),
    ).toHaveAttribute("aria-pressed", "true");
    await user.clear(reasonField());
    reasonField().blur();
    await user.keyboard("e");
    expect(
      screen.getByRole("button", { name: "Needs evidence" }),
    ).toHaveAttribute("aria-pressed", "true");
    expect(reasonField()).toHaveFocus();
    reasonField().blur();
    await user.keyboard("r");
    expect(
      screen.getByRole("button", { name: "Not an issue" }),
    ).toHaveAttribute("aria-pressed", "true");
    expect(reasonField()).toHaveFocus();
    await user.keyboard("Only reachable by admins.");
    await user.keyboard("{Control>}{Enter}{/Control}");
    await waitFor(() =>
      expect(sent("POST", "/reviews/review_new/decisions")).toHaveLength(1),
    );
    expect(
      await bodyOf(sent("POST", "/reviews/review_new/decisions")[0]),
    ).toEqual({
      verdict: "false_positive",
      rationale: "Only reachable by admins.",
    });
  });

  it("leaves single keys alone when shortcuts are off and moves on with J when on", async () => {
    const onNext = vi.fn();
    const first = setup(
      { finding: makeFinding() },
      { shortcuts: false, next: { label: "Next possible issue", onNext } },
    );
    await readyBar();
    await first.user.keyboard("cj");
    expect(
      screen.getByRole("button", { name: "Confirm issue" }),
    ).toHaveAttribute("aria-pressed", "false");
    expect(onNext).not.toHaveBeenCalled();
    first.unmount();

    const second = setup(
      { finding: makeFinding() },
      { next: { label: "Next possible issue", onNext } },
    );
    await decisionBar();
    expect(
      screen.getByRole("button", { name: /Next possible issue/ }),
    ).toHaveAttribute("aria-keyshortcuts", "J");
    await second.user.keyboard("j");
    expect(onNext).toHaveBeenCalledTimes(1);
  });

  it("copies the AI's conclusion into the reason only on request", async () => {
    const { user } = setup({ finding: makeFinding() }, { autoFocus: true });
    const bar = await decisionBar();
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Confirm issue" }),
      ).toHaveFocus(),
    );
    expect(reasonField()).toHaveValue("");
    // The helper text is visible text, not a placeholder that typing hides.
    expect(
      within(bar).getByText("Required. Saved with the decision."),
    ).toBeVisible();
    expect(reasonField()).not.toHaveAttribute("placeholder");
    await user.click(screen.getByRole("button", { name: "Use AI summary" }));
    expect(reasonField()).toHaveValue(
      "Missing object authorization. The order endpoint may read another owner's record.",
    );
    await waitFor(() => expect(reasonField()).toHaveFocus());
    await user.click(screen.getByRole("button", { name: "Use AI summary" }));
    expect(reasonField()).toHaveValue(
      "Missing object authorization. The order endpoint may read another owner's record.",
    );
  });

  it("shows the reason of a duplicate decision on request", async () => {
    const decidedReview = makeTriageReview({
      state: "decided",
      decision: makeDecision({
        verdict: "duplicate",
        duplicateTargetId: "finding_2",
        rationale: "Same handler as finding 2.",
      }),
    });
    delete decidedReview.decision?.severity;
    const { user, sent } = setup({
      finding: makeFinding({
        state: "duplicate",
        duplicateTargetId: "finding_2",
      }),
      decided: [decidedReview],
    });
    const region = await screen.findByRole("region", {
      name: "Current decision",
    });
    expect(within(region).getByText("Duplicate")).toBeVisible();
    expect(within(region).getByTitle("finding_2")).toBeVisible();
    // Only the open-request lookup; the reason is read when asked for.
    await waitFor(() => expect(sent("GET", "/reviews")).toHaveLength(1));
    await user.click(
      within(region).getByRole("button", { name: "Show the reason" }),
    );
    expect(
      await within(region).findByText("Same handler as finding 2."),
    ).toBeVisible();
    expect(within(region).getByText("user_analyst")).toBeVisible();
  });
});
