import { fireEvent, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { fakeServer } from "../../test/fake-server";
import {
  correlationAudit,
  correlationFinding,
  members,
  observation,
  candidates,
  provenance,
} from "../../test/finding-correlation";
import { renderWithAPI } from "../../test/render";
import { failure, json } from "../../test/responses";
import { CorrelationMark } from "./correlation-mark";
import { FindingDuplicates } from "./duplicates";

function show(server = fakeServer()) {
  return renderWithAPI(
    <FindingDuplicates audit={correlationAudit} finding={correlationFinding} />,
    { server },
  );
}
function open(text: string) {
  const summary = screen.getByText(text, { selector: "summary" });
  const details = summary.closest("details")!;
  details.open = true;
  fireEvent(details, new Event("toggle", { bubbles: true }));
}

describe("retained duplicates disclosure", () => {
  it("uses the same system mark for all members without changing independent human state", () => {
    const mark = correlationFinding.correlation;
    if (mark?.availability !== "available")
      throw new Error("Expected an available fixture");
    const first = {
      ...correlationFinding,
      correlation: {
        ...mark,
        automaticDuplicate: false,
      },
    };
    const second = {
      ...correlationFinding,
      correlation: {
        ...mark,
        automaticDuplicate: true,
      },
    };
    renderWithAPI(
      <>
        <CorrelationMark finding={first} />
        <CorrelationMark finding={second} />
      </>,
    );
    expect(screen.getAllByText("Automatic match")).toHaveLength(2);
    expect(first.state).toBe("proposed");
    expect(second.analystVerdict).toBeUndefined();
  });

  it("is lazy, follows every member page and preserves independent decisions and source identity", async () => {
    const server = fakeServer({
      "GET /v1/projects/{projectId}/finding-correlations/{groupKey}/members": ({
        url,
      }) =>
        json(
          url.searchParams.has("cursor")
            ? members([
                observation("third", {
                  state: "confirmed",
                  analystVerdict: "true_positive",
                  analystSeverity: "low",
                }),
              ])
            : members(
                [
                  observation("second", {
                    origin: {
                      ...observation("second").origin,
                      receiptId: correlationFinding.firstProposal.receiptId,
                    },
                  }),
                ],
                true,
              ),
        ),
    });
    const view = show(server);
    expect(view.requests).toHaveLength(0);
    open("Duplicates");
    expect(await screen.findByText("Observation second")).toBeVisible();
    expect(
      screen.getByText(/Same source proposal as this finding/),
    ).toBeVisible();
    expect(screen.getByText(/partial list/)).toBeVisible();
    await view.user.click(
      screen.getByRole("button", { name: "Load more observations" }),
    );
    expect(await screen.findByText("Observation third")).toBeVisible();
    expect(screen.getByText(/complete list/)).toBeVisible();
    expect(screen.getByText("Low")).toBeVisible();
    expect(screen.getAllByText(/AI suggestion: High/)).toHaveLength(2);
    expect(
      screen.getAllByRole("link", { name: "Evidence" })[1],
    ).toHaveAttribute("href", "/issues/audit-right/third?tab=evidence");
    expect(view.requests.every((request) => request.method === "GET")).toBe(
      true,
    );
  });

  it("discards a stale traversal and restarts without repeating the invalid cursor", async () => {
    let head = 0;
    const server = fakeServer({
      "GET /v1/projects/{projectId}/finding-correlations/{groupKey}/members": ({
        url,
      }) =>
        url.searchParams.has("cursor")
          ? failure(409, "finding_correlation_stale", "Membership changed")
          : json(
              members(
                [observation(++head === 1 ? "old" : "fresh")],
                head === 1,
              ),
            ),
    });
    const view = show(server);
    open("Duplicates");
    await screen.findByText("Observation old");
    await view.user.click(
      screen.getByRole("button", { name: "Load more observations" }),
    );
    await screen.findByText("Membership changed");
    expect(screen.queryByText("Observation old")).not.toBeInTheDocument();
    await view.user.click(
      screen.getByRole("button", { name: "Restart from current results" }),
    );
    await screen.findByText("Observation fresh");
    expect(screen.queryByText("Observation old")).not.toBeInTheDocument();
    expect(
      view.requests.filter((request) =>
        new URL(request.url).searchParams.has("cursor"),
      ),
    ).toHaveLength(1);
  });

  it("follows empty candidate work pages and paginates source proposals without promoting weak matches", async () => {
    const weak = observation("weak");
    const server = fakeServer({
      "GET /v1/projects/{projectId}/finding-correlations/{groupKey}/members":
        () => json(members([])),
      "GET /v1/audits/{auditId}/findings/{findingId}/provenance": ({ url }) =>
        json(provenance(!url.searchParams.has("cursor"))),
      "GET /v1/audits/{auditId}/findings/{findingId}/duplicate-candidates": ({
        url,
      }) =>
        json(
          url.searchParams.has("cursor")
            ? candidates([
                {
                  tier: 3,
                  automaticallyCorrelated: false,
                  finding: weak,
                  reasonsTruncated: true,
                  reasons: [
                    {
                      kind: "structural-site",
                      key: "a".repeat(64),
                      sourceReceiptId:
                        correlationFinding.firstProposal.receiptId,
                      targetReceiptId: weak.origin.receiptId,
                      limitations: ["site-is-not-issue-equality"],
                    },
                  ],
                },
              ])
            : candidates([], true),
        ),
    });
    const view = show(server);
    open("Duplicates");
    await screen.findByText(/0 retained observations/);
    open("Possible matches");
    await screen.findByText(/reached its search-work limit/);
    await view.user.click(
      screen.getByRole("button", { name: "Continue searching" }),
    );
    await screen.findByText("Observation weak");
    expect(screen.getByText("Possible match")).toBeVisible();
    expect(screen.getByText(/Only some matching reasons/)).toBeVisible();
    await view.user.click(
      screen.getByRole("button", { name: "Load more source proposals" }),
    );
    open("Search options");
    await screen.findByRole("option", { name: /Supporting proposal 1/ });
    open("Search options");
    await view.user.selectOptions(
      screen.getByLabelText("Source proposal"),
      "receipt-supporting",
    );
    await waitFor(() =>
      expect(
        view.requests.some(
          (request) =>
            new URL(request.url).searchParams.get("sourceReceiptId") ===
            "receipt-supporting",
        ),
      ).toBe(true),
    );
    const last = view.requests.at(-1)!;
    expect(new URL(last.url).searchParams.has("cursor")).toBe(false);
  });

  it("aborts an outstanding request when the finding is left", async () => {
    let signal: AbortSignal | undefined;
    const server = fakeServer({
      "GET /v1/projects/{projectId}/finding-correlations/{groupKey}/members": ({
        request,
      }) => {
        signal = request.signal;
        return new Promise((_resolve, reject) =>
          request.signal.addEventListener("abort", () =>
            reject(new DOMException("Aborted", "AbortError")),
          ),
        );
      },
    });
    const view = show(server);
    open("Duplicates");
    await waitFor(() => expect(signal).toBeDefined());
    view.unmount();
    expect(signal!.aborted).toBe(true);
  });

  it("cancels an earlier source search and clears its cursor when a contribution is selected", async () => {
    let oldSignal: AbortSignal | undefined;
    const server = fakeServer({
      "GET /v1/projects/{projectId}/finding-correlations/{groupKey}/members":
        () => json(members([])),
      "GET /v1/audits/{auditId}/findings/{findingId}/provenance": () =>
        json(provenance()),
      "GET /v1/audits/{auditId}/findings/{findingId}/duplicate-candidates": ({
        request,
        url,
      }) => {
        if (url.searchParams.get("sourceReceiptId") === "receipt-supporting")
          return json(candidates());
        oldSignal = request.signal;
        return new Promise((_resolve, reject) =>
          request.signal.addEventListener("abort", () =>
            reject(new DOMException("Aborted", "AbortError")),
          ),
        );
      },
    });
    const view = show(server);
    open("Duplicates");
    open("Possible matches");
    await waitFor(() => expect(oldSignal).toBeDefined());
    open("Search options");
    await view.user.selectOptions(
      screen.getByLabelText("Source proposal"),
      "receipt-supporting",
    );
    await screen.findByText(/search complete/);
    expect(oldSignal!.aborted).toBe(true);
    expect(new URL(view.requests.at(-1)!.url).searchParams.has("cursor")).toBe(
      false,
    );
  });

  it("explains unavailable keys and keeps possible search accessible", async () => {
    const unavailable = {
      ...correlationFinding,
      correlation: {
        ...correlationFinding.correlation!,
        availability: "unavailable" as const,
        reason: "no_qualified_key" as const,
        automaticDuplicate: false as const,
        hasOtherObservations: false as const,
      },
    };
    // Build the closed unavailable arm rather than carrying available-only fields.
    delete (unavailable.correlation as Record<string, unknown>).groupKey;
    delete (unavailable.correlation as Record<string, unknown>).basis;
    delete (unavailable.correlation as Record<string, unknown>).matchedDomain;
    delete (unavailable.correlation as Record<string, unknown>).representative;
    renderWithAPI(
      <FindingDuplicates audit={correlationAudit} finding={unavailable} />,
    );
    open("Duplicates");
    expect(await screen.findByText(/no qualified issue key/)).toBeVisible();
    expect(
      screen.getByText("Possible matches", { selector: "summary" }),
    ).toBeVisible();
  });
});
