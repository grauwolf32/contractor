import { screen, waitFor } from "@testing-library/react";
import { expect, it } from "vitest";

import { HTTP_EXCHANGE_MEDIA_TYPE } from "../../api/artifacts";
import type { AuditFinding } from "../../api/audits";
import { renderWithAPI } from "../../test/render";
import { EvidencePanel } from "./evidence-panel";
import { issue } from "./test-support";
import { issueReviewFixtures } from "../../test/issue-review-fixtures";
import { fakeServer } from "../../test/fake-server";
import { json, bytes } from "../../test/responses";
import { artifactMetadata } from "../../test/builders";

function finding() {
  const value = issue(
    "audit-test",
    "finding-test",
    "Observed issue",
    "2026-10-13T00:00:00Z",
  );
  value.firstProposal.document.evidence_ids = ["evidence-1"];
  value.firstProposal.evidence = [
    {
      ref: {
        namespace: "finding-proposals",
        name: "http.exchange.proposal-test",
        revision: "exact-source",
      },
      digest: "sha256:" + "a".repeat(64),
      mediaType: HTTP_EXCHANGE_MEDIA_TYPE,
      sizeBytes: 100,
    },
  ];
  value.firstProposal.origin.runId = "run-source";
  value.firstProposal.origin.runDeleted = false;
  value.firstProposal.auditHolds = [];
  return value;
}

function renderPanel(value: AuditFinding) {
  return renderWithAPI(<EvidencePanel finding={value} />);
}

it("links HTTP evidence at its exact Run revision", () => {
  renderPanel(finding());
  expect(
    screen.getByRole("link", { name: "Open HTTP exchange" }),
  ).toHaveAttribute(
    "href",
    "/runs/run-source/artifacts/finding-proposals/http.exchange.proposal-test?revision=exact-source",
  );
});
it("uses the current Audit's retained Project copy after source deletion", () => {
  const value = finding();
  value.firstProposal.origin.runDeleted = true;
  value.firstProposal.auditHolds = [
    {
      auditId: value.auditId,
      projectId: "project-kept",
      createdAt: "2026-10-13T00:00:00Z",
      proposal: value.firstProposal.proposal,
      evidence: [
        {
          ...value.firstProposal.evidence[0]!,
          ref: {
            namespace: "audit-evidence",
            name: "saved-exchange",
            revision: "exact-kept",
          },
        },
      ],
    },
  ];
  renderPanel(value);
  expect(
    screen.getByRole("link", { name: "Open HTTP exchange" }),
  ).toHaveAttribute(
    "href",
    "/projects/project-kept/artifacts/audit-evidence/saved-exchange?revision=exact-kept",
  );
});
it("does not fabricate a link to deleted evidence without a retained copy", () => {
  const value = finding();
  value.firstProposal.origin.runDeleted = true;
  renderPanel(value);
  expect(
    screen.queryByRole("link", { name: "Open HTTP exchange" }),
  ).not.toBeInTheDocument();
  expect(screen.getByText("Source is unavailable.")).toBeVisible();
});

it("loads only two small source snippets even when the finding retains 256 files", async () => {
  const data = issueReviewFixtures();
  const value = data.findings.find(
    (item) => item.findingId === "issue-product-search-sql",
  )!;
  const base = value.firstProposal.evidence[0]!;
  value.firstProposal.evidence = Array.from({ length: 256 }, (_, index) => ({
    ...base,
    ref: { ...base.ref, name: `source-${index}` },
  }));
  value.firstProposal.document.evidence_ids = value.firstProposal.evidence.map(
    (file) => file.ref.name,
  );
  value.firstProposal.auditHolds[0]!.evidence = value.firstProposal.evidence;
  const server = fakeServer({}, (_request, url) => {
    expect(url.pathname).toContain("/projects/project_shop/artifacts/");
    const name = url.pathname
      .replace(/\/metadata$/, "")
      .split("/")
      .at(-1);
    const file = value.firstProposal.evidence.find(
      (item) => item.ref.name === name,
    )!;
    expect(["source-0", "source-1"]).toContain(file.ref.name);
    return url.pathname.endsWith("/metadata")
      ? json(
          artifactMetadata(file.ref.namespace, file.ref.name, {
            artifact: file.ref,
            mediaType: "text/plain",
            size: 12,
          }),
        )
      : bytes("saved source", "text/plain");
  });
  renderWithAPI(<EvidencePanel finding={value} />, { server });
  await waitFor(() =>
    expect(screen.getAllByText("saved source", { exact: true })).toHaveLength(
      2,
    ),
  );
  expect(
    server.requests.filter((request) =>
      new URL(request.url).pathname.includes("/artifacts/"),
    ),
  ).toHaveLength(4);
  expect(screen.getAllByRole("button", { name: "Preview" })).toHaveLength(254);
});
