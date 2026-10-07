import { useQuery } from "@tanstack/react-query";
import { act, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  getAuditReport,
  getAuditReview,
  type AuditReport,
  type AuditReviewAction,
  type AuditReviewRequest,
} from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import { ActionDecision, ReportDecision } from "./review-decision";
import {
  AUDIT_ID,
  failure,
  json,
  makeActionReview,
  makeDecision,
  makeTriageReview,
  proposedReport,
  renderWithServer,
  type Handler,
} from "./test-support";

/** A Server that keeps one review request. */
function reviewServer(
  initial: AuditReviewRequest,
  refuse?: (attempt: number) => Response | undefined,
) {
  const state = { review: initial, attempts: 0 };
  const handle: Handler = async (request, url) => {
    const path = url.pathname;
    const base = `/v1/audits/${AUDIT_ID}/reviews/${state.review.requestId}`;
    if (request.method === "GET" && path === base)
      return json(state.review, 200, { ETag: `"${state.review.revision}"` });
    if (request.method === "POST" && path === `${base}/decisions`) {
      state.attempts += 1;
      const refused = refuse?.(state.attempts);
      if (refused !== undefined) return refused;
      const body = (await request.clone().json()) as {
        action: AuditReviewAction;
        rationale: string;
      };
      const decision = makeDecision({
        requestId: state.review.requestId,
        action: body.action,
        rationale: body.rationale,
      });
      delete decision.verdict;
      delete decision.severity;
      delete decision.findingId;
      state.review = {
        ...state.review,
        state: "decided",
        revision: state.review.revision + 1,
        decision,
      };
      return json({ request: state.review, decision, replayed: false });
    }
    throw new Error(`unexpected ${request.method} ${path}`);
  };
  return { state, handle };
}

/** Reads the request like a page does and renders its decision. */
function Live({
  requestId,
  render,
}: {
  requestId: string;
  render: (review: AuditReviewRequest) => ReactNode;
}) {
  const api = usePublicAPI();
  const review = useQuery({
    queryKey: [...queryKeys.audits.detail(AUDIT_ID), "reviews", requestId],
    queryFn: () => getAuditReview(api, AUDIT_ID, requestId),
  });
  return review.data === undefined ? (
    <p>Loading the request</p>
  ) : (
    <>{render(review.data)}</>
  );
}

function bar() {
  return screen.findByRole("region", { name: "Your decision" });
}

describe("ActionDecision", () => {
  it.each([
    [
      makeActionReview(),
      ["Approve", "Reject"],
      "Approving lets this active test run. Rejecting excludes it from the check.",
    ],
    [
      makeActionReview({
        kind: "requirement-applicability",
        requestedActions: ["approve", "reject", "not_applicable"],
      }),
      ["Approve", "Reject", "Not applicable"],
      "Not applicable settles this requirement with your reason and removes it from the coverage count.",
    ],
    [
      makeActionReview({
        kind: "requirement-applicability",
        requestedActions: ["not_applicable"],
      }),
      ["Not applicable"],
      "Not applicable settles this requirement with your reason and removes it from the coverage count.",
    ],
  ] as const)(
    "offers only the requested actions (%#)",
    async (review, labels, intro) => {
      renderWithServer(
        <ActionDecision auditId={AUDIT_ID} review={review} />,
        () => {
          throw new Error("no request expected");
        },
      );
      const region = await bar();
      expect(
        within(region)
          .getAllByRole("button", { pressed: false })
          .map((button) => button.textContent),
      ).toEqual(labels);
      expect(within(region).getByText(intro)).toBeVisible();
    },
  );

  it("records an approval with the request revision, an idempotency key and the reason", async () => {
    const onDecided = vi.fn();
    const server = reviewServer(makeActionReview());
    const { sent } = renderWithServer(
      <Live
        requestId="review_action"
        render={(review) => (
          <ActionDecision
            auditId={AUDIT_ID}
            review={review}
            onDecided={onDecided}
          />
        )}
      />,
      server.handle,
    );
    const user = userEvent.setup();
    const region = await bar();
    const record = within(region).getByRole("button", {
      name: "Record decision",
    });
    expect(record).toHaveAccessibleDescription("Choose a decision.");
    await user.click(within(region).getByRole("button", { name: "Approve" }));
    expect(record).toHaveAccessibleDescription("Write a short reason.");
    const reason = within(region).getByRole("textbox", { name: "Why" });
    expect(reason).toHaveFocus();
    expect(
      within(region).getByText("Required. Saved with the decision."),
    ).toBeVisible();
    expect(reason).not.toHaveAttribute("placeholder");
    await user.keyboard("   ");
    expect(record).toBeDisabled();
    await user.keyboard("The **target** is our own staging host. ");
    await user.click(record);

    expect(await screen.findByText("Approved")).toBeVisible();
    expect(
      await screen.findByText("target", { selector: "strong" }),
    ).toBeVisible();
    expect(
      screen.queryByRole("region", { name: "Your decision" }),
    ).not.toBeInTheDocument();
    // The refetched request shows the decision; the live region still says
    // what was recorded and focus stays on this decision.
    expect(screen.getByRole("status")).toHaveTextContent(
      "Decision recorded: Approved.",
    );
    expect(
      screen.getByRole("group", { name: "Decision on active test approval" }),
    ).toHaveFocus();
    const posts = sent("POST", "/decisions");
    expect(posts).toHaveLength(1);
    expect(posts[0]?.headers.get("If-Match")).toBe('"2"');
    expect(posts[0]?.headers.get("Idempotency-Key")).toMatch(
      /^audit-action-review-ui-/u,
    );
    expect(await posts[0]?.clone().json()).toEqual({
      action: "approve",
      rationale: "The **target** is our own staging host.",
    });
    expect(onDecided).toHaveBeenCalledTimes(1);
  });

  it("explains a refused decision and does not retry it", async () => {
    const server = reviewServer(makeActionReview(), () =>
      failure(
        412,
        "precondition_failed",
        "resource revision precondition failed",
      ),
    );
    const { sent } = renderWithServer(
      <Live
        requestId="review_action"
        render={(review) => (
          <ActionDecision auditId={AUDIT_ID} review={review} />
        )}
      />,
      server.handle,
    );
    const user = userEvent.setup();
    const region = await bar();
    await user.click(within(region).getByRole("button", { name: "Reject" }));
    await user.keyboard("Out of the agreed scope.");
    await user.click(
      within(region).getByRole("button", { name: "Record decision" }),
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Not saved: this request changed or expired first. The latest version is now shown.",
    );
    await user.click(screen.getByText("Request details"));
    expect(
      screen.getByText("Code precondition_failed · Status 412"),
    ).toBeVisible();
    expect(screen.getByText("Request req_1")).toBeVisible();
    await waitFor(() =>
      expect(sent("GET", "/reviews/review_action").length).toBeGreaterThan(1),
    );
    expect(sent("POST", "/decisions")).toHaveLength(1);
  });

  it("shows a decided or expired request without controls", () => {
    const decided = makeActionReview({
      state: "decided",
      decision: { ...makeDecision({ action: "reject" }) },
    });
    delete decided.decision?.verdict;
    delete decided.decision?.severity;
    const { rerender } = renderWithServer(
      <ActionDecision auditId={AUDIT_ID} review={decided} />,
      () => {
        throw new Error("no request expected");
      },
    );
    expect(screen.getByText("Rejected")).toBeVisible();
    expect(screen.getByText("user_analyst")).toBeVisible();
    expect(
      screen.queryByRole("region", { name: "Your decision" }),
    ).not.toBeInTheDocument();
    rerender(
      <ActionDecision
        auditId={AUDIT_ID}
        review={makeActionReview({ requestId: "other", state: "expired" })}
      />,
    );
    expect(
      screen.getByText("This request expired without a decision."),
    ).toBeVisible();
  });

  describe("where a recorded decision scrolls on its own", () => {
    /** Reports sizes when the test says they changed; jsdom lays nothing out. */
    class FakeResizeObserver implements ResizeObserver {
      static readonly watching = new Set<FakeResizeObserver>();
      readonly #callback: ResizeObserverCallback;
      constructor(callback: ResizeObserverCallback) {
        this.#callback = callback;
      }
      observe() {
        FakeResizeObserver.watching.add(this);
      }
      unobserve() {}
      disconnect() {
        FakeResizeObserver.watching.delete(this);
      }
      static report() {
        for (const observer of FakeResizeObserver.watching)
          observer.#callback([], observer);
      }
    }

    /** Gives `element` these heights and reports the resize. */
    function layOut(
      element: HTMLElement,
      heights: { scrollHeight: number; clientHeight: number },
    ) {
      for (const [name, value] of Object.entries(heights))
        Object.defineProperty(element, name, { configurable: true, value });
      act(() => FakeResizeObserver.report());
    }

    beforeEach(() => {
      vi.stubGlobal("ResizeObserver", FakeResizeObserver);
    });
    afterEach(() => {
      vi.unstubAllGlobals();
      FakeResizeObserver.watching.clear();
    });

    it("is a tab stop while it scrolls, so the keyboard reaches all of a long reason", async () => {
      const decided = makeActionReview({
        state: "decided",
        decision: { ...makeDecision({ action: "approve" }) },
      });
      delete decided.decision?.verdict;
      delete decided.decision?.severity;
      renderWithServer(
        <ActionDecision auditId={AUDIT_ID} review={decided} />,
        () => {
          throw new Error("no request expected");
        },
      );
      const user = userEvent.setup();
      const group = screen.getByRole("group", {
        name: "Decision on active test approval",
      });
      // All of it fits: there is nothing to scroll, so Tab passes it.
      layOut(group, { scrollHeight: 160, clientHeight: 160 });
      expect(group).toHaveAttribute("tabindex", "-1");
      await user.tab();
      expect(group).not.toHaveFocus();
      // A long reason under the pinned footer's cap: Tab reaches the record,
      // and from there the arrow keys scroll it.
      layOut(group, { scrollHeight: 900, clientHeight: 384 });
      expect(group).toHaveAttribute("tabindex", "0");
      await user.tab();
      expect(group).toHaveFocus();
      // Back under the cap (a wider or taller window): no tab stop again.
      layOut(group, { scrollHeight: 300, clientHeight: 300 });
      expect(group).toHaveAttribute("tabindex", "-1");
    });

    it("leaves a pending request's group out of the tab order", () => {
      renderWithServer(
        <ActionDecision auditId={AUDIT_ID} review={makeActionReview()} />,
        () => {
          throw new Error("no request expected");
        },
      );
      const group = screen.getByRole("group", {
        name: "Decision on active test approval",
      });
      // The bar has controls of its own and is never capped.
      layOut(group, { scrollHeight: 900, clientHeight: 384 });
      expect(group).toHaveAttribute("tabindex", "-1");
    });
  });

  it("renders nothing for a request that is not an item action", () => {
    const { container } = renderWithServer(
      <ActionDecision auditId={AUDIT_ID} review={makeTriageReview()} />,
      () => {
        throw new Error("no request expected");
      },
    );
    expect(container).toBeEmptyDOMElement();
  });
});

describe("ReportDecision", () => {
  const reportReview = () =>
    makeActionReview({
      requestId: "review_report",
      subjectKind: "audit-report",
      subjectId: AUDIT_ID,
      kind: "report-acceptance",
      revision: 1,
    });

  it("accepts a proposed report next to its own request", async () => {
    const server = reviewServer(reportReview());
    const { sent } = renderWithServer(
      <Live
        requestId="review_report"
        render={(review) => (
          <ReportDecision
            auditId={AUDIT_ID}
            review={review}
            report={proposedReport(reportReview())}
          />
        )}
      />,
      server.handle,
    );
    const user = userEvent.setup();
    const region = await bar();
    expect(
      within(region).getByText(
        "Approving accepts this report and finishes the check. Rejecting ends the check without an accepted report.",
      ),
    ).toBeVisible();
    expect(
      within(region)
        .getAllByRole("button", { pressed: false })
        .map((button) => button.textContent),
    ).toEqual(["Approve", "Reject"]);
    // Nothing reads as accepted before the Server records an approval.
    expect(screen.queryByText("Approved")).toBeNull();
    expect(screen.getByRole("status")).toBeEmptyDOMElement();

    await user.click(within(region).getByRole("button", { name: "Approve" }));
    await user.keyboard("Coverage and gaps are stated plainly.");
    await user.keyboard("{Control>}{Enter}{/Control}");
    expect(await screen.findByText("Approved")).toBeVisible();
    const [post] = sent("POST", "/reviews/review_report/decisions");
    expect(post?.headers.get("If-Match")).toBe('"1"');
    expect(await post?.clone().json()).toEqual({
      action: "approve",
      rationale: "Coverage and gaps are stated plainly.",
    });
  });

  it.each<[string, AuditReport]>([
    [
      "another request",
      proposedReport(makeActionReview({ requestId: "review_newer" })),
    ],
    ["a report that is no longer proposed", { status: "unavailable" }],
  ])("does not decide a request next to %s", (_, report) => {
    renderWithServer(
      <ReportDecision
        auditId={AUDIT_ID}
        review={reportReview()}
        report={report}
      />,
      () => {
        throw new Error("no request expected");
      },
    );
    expect(
      screen.getByText(
        "This report no longer matches its acceptance request. Load the current report before deciding.",
      ),
    ).toBeVisible();
    expect(
      screen.queryByRole("region", { name: "Your decision" }),
    ).not.toBeInTheDocument();
  });

  describe("next to a report the page reads", () => {
    /** Reads the report and the request like a page, with both refreshed. */
    function ReportPage({
      onRecording,
    }: {
      onRecording: (recording: boolean) => void;
    }) {
      const api = usePublicAPI();
      const report = useQuery({
        queryKey: queryKeys.audits.report(AUDIT_ID),
        queryFn: () => getAuditReport(api, AUDIT_ID),
      });
      const review = useQuery({
        queryKey: [...queryKeys.audits.detail(AUDIT_ID), "reviews", "report"],
        queryFn: () => getAuditReview(api, AUDIT_ID, "review_report"),
      });
      return report.data === undefined || review.data === undefined ? (
        <p>Loading</p>
      ) : (
        <>
          <p>Report {report.data.status}</p>
          <ReportDecision
            auditId={AUDIT_ID}
            review={review.data}
            report={report.data}
            onRecording={onRecording}
          />
        </>
      );
    }

    /**
     * A report that is ready once its request is decided. A refused
     * decision moves the report to `refusedReport`; `afterDecision` answers
     * the request reads that follow a decision (e.g. holds them).
     */
    function reportServer(options: {
      refuse?: () => Response | undefined;
      refusedReport?: AuditReport;
      afterDecision?: () => Promise<void>;
    }) {
      let report: AuditReport = proposedReport(reportReview());
      const ready: AuditReport = { ...report, status: "ready" };
      delete ready.review;
      let sentDecision = false;
      const server = reviewServer(reportReview(), () => {
        const refused = options.refuse?.();
        if (refused !== undefined && options.refusedReport !== undefined)
          report = options.refusedReport;
        return refused;
      });
      const handle: Handler = async (request, url) => {
        const path = url.pathname;
        if (
          request.method === "GET" &&
          path === `/v1/audits/${AUDIT_ID}/report`
        )
          return json(server.state.review.state === "decided" ? ready : report);
        if (request.method === "POST") {
          sentDecision = true;
          return server.handle(request, url);
        }
        if (sentDecision) await options.afterDecision?.();
        return server.handle(request, url);
      };
      return { ...server, handle };
    }

    it("keeps the bar while recording, even once the report moved on", async () => {
      let release: (() => void) | undefined;
      const held = new Promise<void>((resolve) => {
        release = resolve;
      });
      const onRecording = vi.fn();
      const server = reportServer({ afterDecision: () => held });
      renderWithServer(<ReportPage onRecording={onRecording} />, server.handle);
      const user = userEvent.setup();
      const region = await bar();
      const group = screen.getByRole("group", {
        name: "Decision on report acceptance",
      });
      expect(group).not.toHaveClass("ui-footer-record");
      await user.click(within(region).getByRole("button", { name: "Approve" }));
      await user.keyboard("Coverage and gaps are stated plainly.");
      await user.click(
        within(region).getByRole("button", { name: "Record decision" }),
      );
      await waitFor(() => expect(onRecording.mock.calls).toEqual([[true]]));
      // The refresh shows the report ready while the request read is held:
      // the decision is still recording, so its bar stays.
      expect(await screen.findByText("Report ready")).toBeVisible();
      expect(screen.getByRole("region", { name: "Your decision" })).toBe(
        region,
      );
      expect(
        within(region).getByRole("button", { name: "Record decision" }),
      ).toHaveAccessibleDescription("Recording…");
      expect(
        screen.queryByText(/no longer matches its acceptance request/),
      ).toBeNull();
      expect(onRecording.mock.calls).toEqual([[true]]);

      release?.();
      expect(await screen.findByText("Approved")).toBeVisible();
      expect(screen.getByRole("status")).toHaveTextContent(
        "Decision recorded: Approved.",
      );
      expect(
        screen.queryByRole("region", { name: "Your decision" }),
      ).not.toBeInTheDocument();
      // A shown decision is capped where a pane footer is pinned.
      expect(group).toHaveClass("decisions-request", "ui-footer-record");
      await waitFor(() =>
        expect(onRecording.mock.calls).toEqual([[true], [false]]),
      );
    });

    it("explains a refused decision next to a report that no longer matches", async () => {
      const onRecording = vi.fn();
      const server = reportServer({
        refuse: () =>
          failure(
            409,
            "conflict",
            "the review request conflicts with the report",
          ),
        refusedReport: { status: "pending" },
      });
      renderWithServer(<ReportPage onRecording={onRecording} />, server.handle);
      const user = userEvent.setup();
      const region = await bar();
      await user.click(within(region).getByRole("button", { name: "Reject" }));
      await user.keyboard("The summary misses the export endpoint.");
      await user.keyboard("{Control>}{Enter}{/Control}");

      expect(
        await screen.findByText(
          "This report no longer matches its acceptance request. Load the current report before deciding.",
        ),
      ).toBeVisible();
      expect(
        screen.queryByRole("region", { name: "Your decision" }),
      ).not.toBeInTheDocument();
      // The refusal stays explained next to the report that moved on.
      const notice = screen.getByRole("alert");
      expect(notice).toHaveTextContent(
        "Not saved: the decision conflicts with the current state, so nothing was retried.",
      );
      await user.click(within(notice).getByText("Request details"));
      expect(
        within(notice).getByText("Code conflict · Status 409"),
      ).toBeVisible();
      await waitFor(() =>
        expect(onRecording.mock.calls).toEqual([[true], [false]]),
      );
      expect(server.state.attempts).toBe(1);
    });
  });

  describe("without a report from the page", () => {
    /** A Server whose report read waits for `answer`. */
    function reportRead(answer: () => Response | Promise<Response>) {
      const handle: Handler = (request, url) => {
        if (
          request.method === "GET" &&
          url.pathname === `/v1/audits/${AUDIT_ID}/report`
        )
          return answer();
        throw new Error(`unexpected ${request.method} ${url.pathname}`);
      };
      return handle;
    }

    it("reads the report and offers actions only once it carries this request", async () => {
      let release: (() => void) | undefined;
      const read = new Promise<void>((resolve) => {
        release = resolve;
      });
      const { sent } = renderWithServer(
        <ReportDecision auditId={AUDIT_ID} review={reportReview()} />,
        reportRead(async () => {
          await read;
          return json(proposedReport(reportReview()));
        }),
      );
      expect(await screen.findByText("Loading the report…")).toBeVisible();
      expect(
        screen.queryByRole("region", { name: "Your decision" }),
      ).not.toBeInTheDocument();
      release?.();
      const region = await bar();
      expect(
        within(region)
          .getAllByRole("button", { pressed: false })
          .map((button) => button.textContent),
      ).toEqual(["Approve", "Reject"]);
      expect(sent("GET", "/report")).toHaveLength(1);
    });

    it("does not decide next to a report that carries another request", async () => {
      renderWithServer(
        <ReportDecision auditId={AUDIT_ID} review={reportReview()} />,
        reportRead(() =>
          json(proposedReport(makeActionReview({ requestId: "review_newer" }))),
        ),
      );
      expect(
        await screen.findByText(
          "This report no longer matches its acceptance request. Load the current report before deciding.",
        ),
      ).toBeVisible();
      expect(
        screen.queryByRole("region", { name: "Your decision" }),
      ).not.toBeInTheDocument();
    });

    it("explains a report that cannot be read and decides only after it loads", async () => {
      let fail = true;
      const user = userEvent.setup();
      renderWithServer(
        <ReportDecision auditId={AUDIT_ID} review={reportReview()} />,
        reportRead(() =>
          fail
            ? failure(503, "unavailable", "report store unavailable")
            : json(proposedReport(reportReview())),
        ),
      );
      const notice = await screen.findByRole("alert");
      expect(notice).toHaveTextContent(
        "The report could not be loaded, so this request cannot be decided here yet.",
      );
      expect(
        screen.queryByRole("region", { name: "Your decision" }),
      ).not.toBeInTheDocument();
      await user.click(within(notice).getByText("Request details"));
      expect(
        within(notice).getByText("Message: report store unavailable"),
      ).toBeVisible();
      fail = false;
      await user.click(
        within(notice).getByRole("button", { name: "Try again" }),
      );
      expect(await bar()).toBeVisible();
    });

    it("needs no report for a request that is already decided", () => {
      const decided = reportReview();
      decided.state = "decided";
      decided.decision = makeDecision({
        requestId: "review_report",
        action: "reject",
      });
      delete decided.decision.verdict;
      delete decided.decision.severity;
      renderWithServer(
        <ReportDecision auditId={AUDIT_ID} review={decided} />,
        () => {
          throw new Error("no request expected");
        },
      );
      expect(screen.getByText("Rejected")).toBeVisible();
    });
  });
});
