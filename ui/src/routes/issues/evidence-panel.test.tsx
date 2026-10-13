import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { expect, it } from "vitest";
import { HTTP_EXCHANGE_MEDIA_TYPE } from "../../api/artifacts";
import { issue } from "./test-support";
import { EvidencePanel } from "./evidence-panel";

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
it("links HTTP evidence at its exact Run revision", () => {
  render(
    <MemoryRouter>
      <EvidencePanel finding={finding()} />
    </MemoryRouter>,
  );
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
  render(
    <MemoryRouter>
      <EvidencePanel finding={value} />
    </MemoryRouter>,
  );
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
  render(
    <MemoryRouter>
      <EvidencePanel finding={value} />
    </MemoryRouter>,
  );
  expect(
    screen.queryByRole("link", { name: "Open HTTP exchange" }),
  ).not.toBeInTheDocument();
  expect(screen.getByText("Source is unavailable.")).toBeVisible();
});
