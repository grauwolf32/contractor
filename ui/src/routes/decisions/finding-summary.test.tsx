import { screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { AuditFinding } from "../../api/audits";
import { FindingSummary } from "./finding-summary";
import {
  AUDIT_ID,
  DIGEST,
  json,
  makeFinding,
  NOW,
  renderWithServer,
} from "./test-support";

const DESCRIPTION = [
  "The handler reads any order by ID.",
  "",
  "It never compares `ownerId` with the caller.",
  "",
  "## Impact",
  "",
  "Any signed-in user can read **all** orders.",
  "",
  "## Fix",
  "",
  "Scope the query to the caller.",
].join("\n");

function detailedFinding(overrides: Partial<AuditFinding> = {}): AuditFinding {
  const finding = makeFinding(
    {
      currentAssessment: {
        assessmentId: "assessment_1",
        semanticAssessment: "supported",
        result: {
          ref: { namespace: "audit-results", name: "check-1", revision: "r2" },
          digest: DIGEST,
        },
        receiptId: "receipt_1",
        directVerification: false,
        acceptedAt: NOW,
      },
      ...overrides,
    },
    {
      title: "Any user can read any order",
      description: DESCRIPTION,
      subject: { kind: "openapi-operation", key: "GET /orders/{id}" },
      standard_refs: [
        { scheme: "CWE", version: "4.20", requirement_id: "CWE-639" },
        { scheme: "owasp-web-top10", version: "2021", requirement_id: "A01" },
      ],
      limitations: ["How tokens are issued is not visible in this code."],
      preconditions: [
        "The caller has an account.",
        "Order IDs are sequential.",
      ],
      evidence_ids: ["evidence-1"],
      locations: [
        { file: "orders/views.py", line: 42 },
        { file: "orders/views.py", range: { start_line: 10, end_line: 12 } },
        { url: "https://shop.example/orders/1", method: "GET" },
        { file: "orders/serializers.py", line: 7 },
        { file: "orders/urls.py", line: 3 },
      ],
      http_exchange: {
        request_id: 7,
        request_tag: "probe",
        attempts: [
          {
            method: "GET",
            url: "https://shop.example/orders/1",
            headers: [{ name: "Accept", value: "application/json" }],
            body_base64: "",
            status: 200,
          },
        ],
      },
    },
  );
  finding.firstProposal.evidence = [
    {
      ref: {
        namespace: "audit-evidence",
        name: "response-body",
        revision: "r1",
      },
      digest: DIGEST,
      mediaType: "application/json",
      sizeBytes: 2048,
    },
  ];
  return finding;
}

function noRequests(): Response {
  throw new Error("no request expected");
}

/** The value of a fact (dt/dd pair) as text. */
function fact(name: string): string | null | undefined {
  return screen.getByText(name, { selector: "dt" }).nextElementSibling
    ?.textContent;
}

describe("FindingSummary", () => {
  it("maps every section to the proposal's own fields", async () => {
    renderWithServer(
      <FindingSummary auditId={AUDIT_ID} finding={detailedFinding()} />,
      noRequests,
    );
    const article = screen.getByRole("article", {
      name: "Any user can read any order",
    });
    expect(
      within(article).getByRole("heading", {
        level: 2,
        name: "Any user can read any order",
      }),
    ).toBeVisible();
    // A possible issue reads as one, apart from confirmed issues.
    expect(within(article).getByText("Needs review")).toBeVisible();
    expect(within(article).getByText("Possible issue")).toBeVisible();

    expect(fact("Endpoint")).toBe("GET /orders/{id}");
    expect(
      screen.getByText("Endpoint", { selector: "dt" }).nextElementSibling
        ?.firstElementChild,
    ).toHaveClass("ui-method-chip");
    expect(fact("Weakness")).toBe("CWE-639");
    expect(fact("Standard")).toBe("A01 owasp-web-top10@2021");
    expect(fact("Severity")).toBe("Not set · AI suggestion: High");
    expect(fact("Verification")).toBe("Supported");

    const found = screen.getByRole("region", { name: "What the AI found" });
    expect(
      await within(found).findByText("The handler reads any order by ID."),
    ).toBeVisible();
    expect(
      within(found).getByText("ownerId", { selector: "code" }),
    ).toBeVisible();
    expect(
      within(found).getByText("Scope the query to the caller."),
    ).toBeVisible();
    expect(within(found).queryByText(/Any signed-in user/)).toBeNull();

    const impact = screen.getByRole("region", { name: "Impact" });
    expect(
      await within(impact).findByText("all", { selector: "strong" }),
    ).toBeVisible();

    const locations = screen.getByRole("region", { name: "Finding locations" });
    expect(within(locations).getByText("orders/views.py:42")).toBeVisible();
    expect(within(locations).getByText("orders/views.py:10–12")).toBeVisible();
    expect(
      within(locations).getByText("https://shop.example/orders/1"),
    ).toBeVisible();
    expect(within(locations).getAllByRole("listitem")).toHaveLength(5);
    expect(within(locations).getByText(/Captured HTTP evidence/)).toBeVisible();

    const unsure = screen.getByRole("region", {
      name: "What the AI is unsure about",
    });
    expect(
      within(unsure).getByText(
        "How tokens are issued is not visible in this code.",
      ),
    ).toBeVisible();
    expect(within(unsure).getByText("It depends on:")).toBeVisible();
    expect(
      within(unsure)
        .getAllByRole("listitem")
        .map((item) => item.textContent),
    ).toEqual(["The caller has an account.", "Order IDs are sequential."]);

    const evidence = screen.getByRole("region", { name: "Evidence" });
    const row = within(evidence).getByRole("listitem");
    expect(row).toHaveTextContent("evidence-1");
    expect(row).toHaveTextContent("audit-evidence/response-body");
    expect(row).toHaveTextContent("application/json · 2.0 KiB");
  });

  it("keeps a compact preview to the first paragraph, three locations and the doubts", async () => {
    renderWithServer(
      <FindingSummary
        auditId={AUDIT_ID}
        finding={detailedFinding()}
        variant="compact"
        titleAs="h3"
      />,
      noRequests,
    );
    expect(
      screen.getByRole("heading", {
        level: 3,
        name: "Any user can read any order",
      }),
    ).toBeVisible();
    const found = screen.getByRole("region", { name: "What the AI found" });
    expect(
      within(found).getByRole("heading", {
        level: 4,
        name: "What the AI found",
      }),
    ).toBeVisible();
    expect(
      await within(found).findByText("The handler reads any order by ID."),
    ).toBeVisible();
    expect(within(found).queryByText(/never compares/)).toBeNull();
    const locations = screen.getByRole("region", { name: "Finding locations" });
    expect(within(locations).getAllByRole("listitem")).toHaveLength(3);
    expect(screen.getByText("2 more locations")).toBeVisible();
    expect(screen.queryByText(/Captured HTTP evidence/)).toBeNull();
    expect(screen.queryByRole("region", { name: "Impact" })).toBeNull();
    expect(screen.queryByRole("region", { name: "Evidence" })).toBeNull();
    expect(
      screen.getByRole("region", { name: "What the AI is unsure about" }),
    ).toBeVisible();
    expect(fact("Severity")).toBe("Not set · AI suggestion: High");
  });

  it("leaves out sections without data and never shows a suggestion as a rating", () => {
    renderWithServer(
      <FindingSummary
        auditId={AUDIT_ID}
        finding={makeFinding({}, { description: "", severity_suggestion: "" })}
      />,
      noRequests,
    );
    expect(
      screen.getByRole("heading", { name: "Missing object authorization" }),
    ).toBeVisible();
    expect(fact("Severity")).toBe("Not set");
    for (const name of [
      "What the AI found",
      "Impact",
      "What the AI is unsure about",
      "Evidence",
      "Finding locations",
    ])
      expect(screen.queryByRole("region", { name })).toBeNull();
    for (const term of ["Endpoint", "Subject", "Weakness", "Verification"])
      expect(screen.queryByText(term, { selector: "dt" })).toBeNull();
  });

  it("reads a confirmed finding as an issue with the analyst's severity", () => {
    renderWithServer(
      <FindingSummary
        auditId={AUDIT_ID}
        finding={makeFinding(
          {
            state: "confirmed",
            analystVerdict: "true_positive",
            analystSeverity: "critical",
          },
          { subject: { kind: "component", key: "orders" } },
        )}
      />,
      noRequests,
    );
    expect(screen.getByText("Confirmed")).toBeVisible();
    expect(screen.getByText("Issue")).toBeVisible();
    expect(screen.queryByText("Possible issue")).toBeNull();
    expect(fact("Severity")).toBe("Critical");
    expect(fact("Subject")).toBe("orders component");
  });

  it("names the original of a duplicate", async () => {
    const original = makeFinding(
      { findingId: "finding_original" },
      { title: "Orders are readable by anyone" },
    );
    renderWithServer(
      <FindingSummary
        auditId={AUDIT_ID}
        finding={makeFinding({
          state: "duplicate",
          duplicateTargetId: "finding_original",
        })}
      />,
      (_request, url) => {
        if (url.pathname === `/v1/audits/${AUDIT_ID}/findings/finding_original`)
          return json(original, 200, { ETag: `"${original.revision}"` });
        throw new Error(`unexpected ${url.pathname}`);
      },
    );
    expect(
      await screen.findByText(/Orders are readable by anyone/),
    ).toBeVisible();
    expect(fact("Duplicate of")).toContain("finding_original");
  });
});
