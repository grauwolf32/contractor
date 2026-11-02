import { screen, waitFor } from "@testing-library/react";
import { expect, it } from "vitest";
import { HTTP_EXCHANGE_MEDIA_TYPE } from "../../api/artifacts";
import { renderWithAPI } from "../../test/render";
import { EvidencePanel, SourceEvidence } from "./evidence-panel";
import { issueReviewFixtures } from "../../test/issue-review-fixtures";
import { fakeServer } from "../../test/fake-server";
import { json, bytes } from "../../test/responses";
import { artifactMetadata, finding, findingSource } from "../../test/builders";

it.each([false, true])(
  "always opens the exact retained Project evidence (source deleted=%s)",
  (runDeleted) => {
    const value = finding("project-kept", "finding-test");
    const file = {
      ref: {
        namespace: "finding-materials",
        name: "saved-exchange",
        revision: "exact-kept",
      },
      digest: "sha256:" + "a".repeat(64),
      mediaType: HTTP_EXCHANGE_MEDIA_TYPE,
      sizeBytes: 100,
    };
    const source = findingSource(
      value,
      { projectEvidence: [file] },
      { evidence_ids: ["evidence-1"] },
    );
    source.origin.runDeleted = runDeleted;
    renderWithAPI(<SourceEvidence source={source} />);
    expect(
      screen.getByRole("link", { name: "Open HTTP exchange" }),
    ).toHaveAttribute(
      "href",
      "/projects/project-kept/artifacts/finding-materials/saved-exchange?revision=exact-kept",
    );
  },
);
it("reads a separate immutable source receipt for the canonical Finding", async () => {
  const value = finding("project", "finding");
  const source = findingSource(
    value,
    {},
    { locations: [{ file: "orders.py", line: 42 }] },
  );
  const server = fakeServer({
    "GET /v1/projects/{projectId}/finding-sources/{receiptId}": ({
      params,
    }) => {
      expect(params).toEqual({
        projectId: "project",
        receiptId: value.firstReceiptId,
      });
      return json(source);
    },
  });
  renderWithAPI(<EvidencePanel finding={value} />, { server });
  expect(await screen.findByText("orders.py:42")).toBeVisible();
  expect(server.requests).toHaveLength(1);
});
it("loads only two small source snippets when the source retains 256 files", async () => {
  const data = issueReviewFixtures();
  const source = data.sources.find(
    (item) => item.findingId === "issue-product-search-sql",
  )!;
  const base = source.projectEvidence![0]!;
  source.projectEvidence = Array.from({ length: 256 }, (_, index) => ({
    ...base,
    ref: { ...base.ref, name: `source-${index}` },
  }));
  source.document.evidence_ids = source.projectEvidence.map(
    (file) => file.ref.name,
  );
  const server = fakeServer({}, (_request, url) => {
    expect(url.pathname).toContain("/projects/project_shop/artifacts/");
    const name = url.pathname
      .replace(/\/metadata$/, "")
      .split("/")
      .at(-1)!;
    const file = source.projectEvidence!.find(
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
  renderWithAPI(<SourceEvidence source={source} />, { server });
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
